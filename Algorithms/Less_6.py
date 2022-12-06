from collections import deque


class TreeNode:

    def __init__(self, parent, key: int):
        """
        Creates a node.

        Args:
            parent: The node's parent.
            key: key of the node.
        """
        self.key = key
        self.left = None
        self.right = None
        self.parent = parent

    def insert(self, node) -> None:
        """
        Inserting an element into a tree

        node: The tree where we want to insert the element
        """
        if node is None:
            return
        if node.key < self.key:
            if self.left is None:
                node.parent = self
                self.left = node
            else:
                self.left.insert(node)
        else:
            if self.right is None:
                node.parent = self
                self.right = node
            else:
                self.right.insert(node)

    def breadth_first(self) -> [list, list]:
        """
        Bypass in width (BFS)

        :return: elements & levels
        """
        key_level = {}
        q = deque()
        q.append((self, 0))
        while q:
            node, level = q.popleft()
            key_level[node.key] = level
            if node.left:
                q.append((node.left, level + 1))
            if node.right:
                q.append((node.right, level + 1))

        return key_level.keys(), key_level.values()

    def find(self, k):
        """
        Node search by key

        k: key
        :return: node
        """
        if self.key == k:
            return self
        elif k < self.key:
            if self.left is None:
                return None
            else:
                return self.left.find(k)
        else:
            if self.right is None:
                return None
            else:
                return self.right.find(k)

    def pre_order(self, pre_order_list: list = None) -> list:
        """
        Depth walk (DFS)

        pre_order_list: list of tree elements.
        :return: pre_order_list
        """
        if pre_order_list is None:
            pre_order_list = list()

        if self:
            pre_order_list.append(self.key)
            if self.left:
                self.left.pre_order(pre_order_list)
            if self.right:
                self.right.pre_order(pre_order_list)

        if self.parent is None:
            return pre_order_list

    def left_rotate(self):
        """
        Left rotation of the tree

        :return: node
        """
        if self is None or self.right is None:
            return self
        parent = self.parent
        right = self.right
        right_left = right.left

        if parent:
            if parent.left == self:
                parent.left = right
            else:
                parent.right = right
        right.parent = parent

        right.left = self
        self.parent = right

        self.right = right_left
        if right_left:
            right_left.parent = self

        return right


def build_tree(keys: list) -> TreeNode:
    """
    Building a tree from the received list

    keys: list of elements
    :return: TreeNode
    """
    root = TreeNode(None, keys[0])
    if len(keys) != 1:
        for key in keys[1:]:
            root.insert(TreeNode(None, key))

    return root


def get_level_order_keys_and_levels(preorder_keys: list) -> [list, list]:
    """
    Function to get a list of tree elements and their levels

    preorder_keys: list of elements
    :return: list of tree elements and their levels
    """
    root = build_tree(preorder_keys)
    if len(preorder_keys) != 1:
        return root.breadth_first()
    else:
        return [root.key], [0]


def find_and_left_rotate(preorder_keys: list, key: int) -> list:
    """
    The function finds a node by key and rotates it if possible

    preorder_keys: list of elements
    key: key node
    :return: list
    """
    root = build_tree(preorder_keys)
    node = root.find(key)
    node = node.left_rotate()
    return root.pre_order() or node.pre_order()


def solution_1():
    preorder_keys = list(map(int, input().split()))
    level_order_keys, levels = get_level_order_keys_and_levels(preorder_keys)
    print(' '.join(map(str, level_order_keys)))
    print(' '.join(map(str, levels)))


def solution_2():
    preorder_keys = list(map(int, input().split()))
    key = int(input())
    preorder_keys_after_rotate = find_and_left_rotate(preorder_keys, key)
    print(' '.join(map(str, preorder_keys_after_rotate)))


if __name__ == '__main__':
    solution_1()
    solution_2()
