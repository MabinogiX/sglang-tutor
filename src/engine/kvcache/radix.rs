use std::{
    cell::RefCell,
    collections::{BTreeSet, HashMap},
    rc::Rc,
};

use super::{BaseCacheHandle, CacheManager, KVCacheError, KVCachePool, Result};

type NodeId = usize;

/// One token in a prefix-sharing radix tree.
#[derive(Debug, Clone)]
pub struct RadixNode {
    pub token: i64,
    pub ref_count: usize,
    pub page_id: Option<usize>,
    depth: isize,
    parent: Option<NodeId>,
    children: HashMap<i64, NodeId>,
}

impl RadixNode {
    fn root() -> Self {
        Self {
            token: -1,
            ref_count: 1,
            page_id: None,
            depth: -1,
            parent: None,
            children: HashMap::new(),
        }
    }

    fn child(token: i64, parent: NodeId, depth: isize) -> Self {
        Self {
            token,
            ref_count: 0,
            page_id: None,
            depth,
            parent: Some(parent),
            children: HashMap::new(),
        }
    }
}

/// Prefix cache with page-granular eviction.
///
/// `Rc<RefCell<_>>` lets the scheduler retain access to the same pool while the
/// cache manager returns pages during eviction. Scheduling remains single-threaded
/// at this stage; multi-threaded scheduling should replace it with `Arc<Mutex<_>>`.
pub struct RadixCacheManager {
    pool: Rc<RefCell<KVCachePool>>,
    page_size: usize,
    nodes: Vec<RadixNode>,
}

impl RadixCacheManager {
    pub fn new(pool: Rc<RefCell<KVCachePool>>, page_size: usize) -> Result<Self> {
        if page_size == 0 {
            return Err(KVCacheError::InvalidArgument(
                "page_size must be greater than zero".to_owned(),
            ));
        }
        Ok(Self {
            pool,
            page_size,
            nodes: vec![RadixNode::root()],
        })
    }

    fn node(&self, id: NodeId) -> &RadixNode {
        &self.nodes[id]
    }

    fn node_mut(&mut self, id: NodeId) -> &mut RadixNode {
        &mut self.nodes[id]
    }

    fn child_id(&self, node: NodeId, token: i64) -> Option<NodeId> {
        self.node(node).children.get(&token).copied()
    }

    fn get_or_create_child(&mut self, parent: NodeId, token: i64) -> NodeId {
        if let Some(child) = self.child_id(parent, token) {
            return child;
        }

        let child = self.nodes.len();
        let depth = self.node(parent).depth + 1;
        self.nodes.push(RadixNode::child(token, parent, depth));
        self.node_mut(parent).children.insert(token, child);
        child
    }

    fn required_pages(&self, input_ids: &[i64]) -> usize {
        input_ids.len().div_ceil(self.page_size)
    }

    fn ensure_page_table(&self, input_ids: &[i64], handle: &BaseCacheHandle) -> Result<()> {
        let required = self.required_pages(input_ids);
        if handle.page_ids.len() < required {
            return Err(KVCacheError::InvalidArgument(format!(
                "page table has {} pages but {required} are required for {} tokens",
                handle.page_ids.len(),
                input_ids.len()
            )));
        }
        Ok(())
    }

    /// Return the page-aligned reusable prefix. The final token is always
    /// excluded so the model recomputes logits for sampling.
    pub fn match_prefix(&self, input_ids: &[i64]) -> (usize, Vec<usize>) {
        if input_ids.is_empty() {
            return (0, Vec::new());
        }

        let mut node = 0;
        let mut path = Vec::new();
        for &token in input_ids {
            let Some(child) = self.child_id(node, token) else {
                break;
            };
            node = child;
            path.push(node);
        }

        let matched_len = (path.len() / self.page_size) * self.page_size;
        let maximum = ((input_ids.len() - 1) / self.page_size) * self.page_size;
        let matched_len = matched_len.min(maximum);
        let shared_pages = (0..matched_len / self.page_size)
            .filter_map(|page| self.node(path[(page + 1) * self.page_size - 1]).page_id)
            .collect();
        (matched_len, shared_pages)
    }

    /// Claim a reference to every node of a prompt path.
    pub fn insert(&mut self, input_ids: &[i64], handle: &BaseCacheHandle) -> Result<()> {
        self.ensure_page_table(input_ids, handle)?;
        let mut node = 0;
        for (depth, &token) in input_ids.iter().enumerate() {
            node = self.get_or_create_child(node, token);
            let page_id = handle.page_ids[depth / self.page_size];
            let node = self.node_mut(node);
            node.ref_count += 1;
            node.page_id = Some(page_id);
        }
        Ok(())
    }

    /// Undo an insert whose forward pass failed before writing its KV values.
    pub fn rollback_insert(&mut self, input_ids: &[i64], handle: &BaseCacheHandle) -> Result<()> {
        let mut node = 0;
        let mut path = Vec::new();
        for &token in input_ids {
            let Some(child) = self.child_id(node, token) else {
                break;
            };
            path.push(child);
            node = child;
        }

        for &node in &path {
            self.node_mut(node).ref_count = self.node(node).ref_count.saturating_sub(1);
        }

        let mut freed_pages = BTreeSet::new();
        for &node in path.iter().rev() {
            let removable = self.node(node).ref_count == 0 && self.node(node).children.is_empty();
            if !removable {
                continue;
            }
            if let Some(page_id) = self.node(node).page_id {
                freed_pages.insert(page_id);
            }
            let parent = self
                .node(node)
                .parent
                .expect("root is never on an inserted path");
            let token = self.node(node).token;
            self.node_mut(parent).children.remove(&token);
        }
        if !freed_pages.is_empty() {
            self.pool
                .borrow_mut()
                .free_pages_by_id(freed_pages.into_iter());
        }

        let used = self.required_pages(input_ids);
        self.pool.borrow_mut().free_pages_by_id(
            handle.page_ids[used.min(handle.page_ids.len())..]
                .iter()
                .copied(),
        );
        Ok(())
    }

    /// Drop request references, preserve written KV in the tree, and return
    /// over-allocated pages that cannot contain a token.
    pub fn remove(&mut self, input_ids: &[i64], handle: &BaseCacheHandle) -> Result<()> {
        let mut node = 0;
        for (depth, &token) in input_ids.iter().enumerate() {
            let child = if let Some(child) = self.child_id(node, token) {
                self.node_mut(child).ref_count = self.node(child).ref_count.saturating_sub(1);
                child
            } else {
                let child = self.get_or_create_child(node, token);
                if depth < handle.page_ids.len() * self.page_size {
                    self.node_mut(child).page_id = Some(handle.page_ids[depth / self.page_size]);
                }
                child
            };
            node = child;
        }

        let used = self.required_pages(input_ids);
        self.pool.borrow_mut().free_pages_by_id(
            handle.page_ids[used.min(handle.page_ids.len())..]
                .iter()
                .copied(),
        );
        Ok(())
    }

    /// Evict unreferenced leaf chains. It only begins a page when the entire
    /// page can be detached, so a shared page is never partially released.
    pub fn evict(&mut self, num_pages: usize) -> Vec<usize> {
        let mut freed = Vec::new();
        let mut remaining = num_pages;

        while remaining > 0 {
            let mut leaves: Vec<_> = self
                .iter_leaves()
                .into_iter()
                .filter(|&node| self.node(node).ref_count == 0)
                .collect();
            if leaves.is_empty() {
                break;
            }
            leaves.sort_unstable_by_key(|&node| std::cmp::Reverse(self.node(node).depth));

            let mut progress = false;
            for leaf in leaves {
                let count = self.evict_leaf_chain(leaf, remaining, &mut freed);
                if count > 0 {
                    remaining -= count;
                    progress = true;
                    break;
                }
            }
            if !progress {
                break;
            }
        }
        freed
    }

    fn iter_leaves(&self) -> Vec<NodeId> {
        let mut leaves = Vec::new();
        let mut stack = vec![0];
        while let Some(node) = stack.pop() {
            let node_ref = self.node(node);
            if node != 0 && node_ref.children.is_empty() {
                leaves.push(node);
            }
            stack.extend(node_ref.children.values().copied());
        }
        leaves
    }

    fn evict_leaf_chain(
        &mut self,
        leaf: NodeId,
        max_pages: usize,
        freed: &mut Vec<usize>,
    ) -> usize {
        let mut chain = Vec::new();
        let mut node = leaf;
        while node != 0 && self.node(node).ref_count == 0 && self.node(node).children.len() <= 1 {
            chain.push(node);
            node = self
                .node(node)
                .parent
                .expect("non-root nodes have a parent");
        }

        let depth = self.node(leaf).depth as usize;
        let mut position = 0;
        let mut pages_freed = 0;
        while position < chain.len() && pages_freed < max_pages {
            let group_size = if position == 0 {
                depth % self.page_size + 1
            } else {
                self.page_size
            };
            if position + group_size > chain.len() {
                break;
            }

            let owner = chain[position];
            for &victim in &chain[position..position + group_size] {
                let parent = self.node(victim).parent.expect("victim is not root");
                let token = self.node(victim).token;
                self.node_mut(parent).children.remove(&token);
            }
            if let Some(page_id) = self.node(owner).page_id {
                self.pool.borrow_mut().free_pages_by_id([page_id]);
                freed.push(page_id);
                pages_freed += 1;
            }
            position += group_size;
        }
        pages_freed
    }
}

impl CacheManager for RadixCacheManager {
    fn match_prefix(&self, input_ids: &[i64]) -> Result<(usize, Vec<usize>)> {
        Ok(self.match_prefix(input_ids))
    }

    fn insert(&mut self, input_ids: &[i64], handle: &BaseCacheHandle) -> Result<()> {
        self.insert(input_ids, handle)
    }

    fn evict(&mut self, num_pages: usize) -> Result<Vec<usize>> {
        Ok(self.evict(num_pages))
    }

    fn remove(&mut self, input_ids: &[i64], handle: &BaseCacheHandle) -> Result<()> {
        self.remove(input_ids, handle)
    }

    fn rollback_insert(&mut self, input_ids: &[i64], handle: &BaseCacheHandle) -> Result<()> {
        self.rollback_insert(input_ids, handle)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::kvcache::KVCacheLayout;

    fn cache(num_pages: usize) -> (Rc<RefCell<KVCachePool>>, RadixCacheManager) {
        let pool = Rc::new(RefCell::new(KVCachePool::without_tensor(
            KVCacheLayout::new(2, num_pages, 4, 4, 32).unwrap(),
        )));
        let manager = RadixCacheManager::new(Rc::clone(&pool), 4).unwrap();
        (pool, manager)
    }

    #[test]
    fn matches_complete_pages_but_recomputes_the_last_token() {
        let (pool, mut cache) = cache(20);
        let handle = pool.borrow_mut().alloc(2).unwrap();
        cache.insert(&[1, 2, 3, 4, 5, 6, 7, 8], &handle).unwrap();

        assert_eq!(
            cache.match_prefix(&[1, 2, 3, 4, 9, 0]),
            (4, vec![handle.page_ids[0]])
        );
        assert_eq!(
            cache.match_prefix(&[1, 2, 3, 4, 5, 6, 7, 8]),
            (4, vec![handle.page_ids[0]])
        );
    }

    #[test]
    fn removes_then_evicts_full_and_partial_pages() {
        let (pool, mut cache) = cache(20);
        let handle = pool.borrow_mut().alloc(2).unwrap();
        cache.insert(&[1, 2, 3, 4, 5, 6], &handle).unwrap();
        cache.remove(&[1, 2, 3, 4, 5, 6], &handle).unwrap();

        let mut evicted = cache.evict(2);
        evicted.sort_unstable();
        let mut expected = handle.page_ids.clone();
        expected.sort_unstable();
        assert_eq!(evicted, expected);
        assert_eq!(pool.borrow().free_count(), 20);
    }

    #[test]
    fn preserves_shared_prefix_when_one_insert_is_rolled_back() {
        let (pool, mut cache) = cache(30);
        let first = pool.borrow_mut().alloc(2).unwrap();
        cache.insert(&[1, 2, 3, 4, 5, 6, 7, 8], &first).unwrap();
        let (_, shared) = cache.match_prefix(&[1, 2, 3, 4, 5, 6, 7, 8, 9]);
        let mut second = pool.borrow_mut().alloc(1).unwrap();
        second.page_ids.splice(0..0, shared.iter().copied());
        second.num_shared = shared.len();
        cache
            .insert(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], &second)
            .unwrap();

        cache
            .rollback_insert(&[1, 2, 3, 4, 5, 6, 7, 8], &first)
            .unwrap();
        assert_eq!(
            cache.match_prefix(&[1, 2, 3, 4, 5, 6, 7, 8, 9]),
            (8, shared)
        );
        assert_eq!(pool.borrow().free_count(), 27);
    }

    #[test]
    fn rollback_returns_unshared_pages_and_removes_the_prefix() {
        let (pool, mut cache) = cache(20);
        let handle = pool.borrow_mut().alloc(2).unwrap();
        cache.insert(&[1, 2, 3, 4, 5, 6, 7, 8], &handle).unwrap();
        cache
            .rollback_insert(&[1, 2, 3, 4, 5, 6, 7, 8], &handle)
            .unwrap();

        assert_eq!(cache.match_prefix(&[1, 2, 3, 4]), (0, vec![]));
        assert_eq!(pool.borrow().free_count(), 20);
    }

    #[test]
    fn shared_prefix_pages_are_evicted_exactly_once() {
        let (pool, mut cache) = cache(20);
        let first = pool.borrow_mut().alloc(2).unwrap();
        cache.insert(&[1, 2, 3, 4, 5, 6], &first).unwrap();

        let (_, shared) = cache.match_prefix(&[1, 2, 3, 4, 7, 8]);
        let mut second = pool.borrow_mut().alloc(1).unwrap();
        second.page_ids.splice(0..0, shared.iter().copied());
        cache.insert(&[1, 2, 3, 4, 7, 8], &second).unwrap();
        cache.remove(&[1, 2, 3, 4, 5, 6], &first).unwrap();
        cache.remove(&[1, 2, 3, 4, 7, 8], &second).unwrap();

        let mut evicted = cache.evict(10);
        evicted.sort_unstable();
        let mut expected = vec![first.page_ids[0], first.page_ids[1], second.page_ids[1]];
        expected.sort_unstable();
        assert_eq!(evicted, expected);
        assert_eq!(pool.borrow().free_count(), 20);
    }

    #[test]
    fn referenced_nodes_cannot_be_evicted() {
        let (pool, mut cache) = cache(20);
        let handle = pool.borrow_mut().alloc(1).unwrap();
        cache.insert(&[1, 2, 3, 4], &handle).unwrap();

        assert!(cache.evict(1).is_empty());
        assert_eq!(pool.borrow().free_count(), 19);
    }
}
