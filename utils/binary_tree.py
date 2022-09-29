from __future__ import annotations

from typing import Optional, List, Tuple, Dict

import numpy as np
from binarytree import _build_tree_string, _get_tree_properties

from utils.constants import LABELS_MAP


class Node:
    def __init__(self, value, left: Optional[Node] = None, right: Optional[Node] = None):
        self.val = self.value = value
        self.left = left
        self.right = right

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, Node):
            return False
        if (self.left is None and o.left is not None) or (o.left is None and self.left is not None):
            return False
        if (self.right is None and o.right is not None) or (o.right is None and self.right is not None):
            return False
        val_eq = np.all(self.val == o.val)
        if self.is_leaf():
            return val_eq
        left_eq = True if self.left is None else self.left == o.left
        right_eq = True if self.right is None else self.right == o.right
        return val_eq and left_eq and right_eq

    @classmethod
    def init_from_dict(cls, code_dict: Dict[int, Tuple[int, ...]]) -> Node:
        root = cls([])
        current = root
        for val, code in code_dict.items():
            current.value.append(val)
            for c in code:
                if c == 0:
                    if current.left is None:
                        current.left = cls([])
                    current = current.left
                else:  # c == 1
                    if current.right is None:
                        current.right = cls([])
                    current = current.right
                current.value.append(val)
            current = root
        return root

    @classmethod
    def init_from_dict_extended(cls, code_dict: Dict[int, Tuple[int, ...]], root=None) -> Node:
        if root is None:
            root = cls([])
        for val, code in code_dict.items():
            root.value.append(val)
            if len(code):
                c = code[0]
                if c < 1:
                    if root.left is None:
                        root.left = cls([])
                    cls.init_from_dict_extended({val: code[1:]}, root.left)
                if c > 0:
                    if root.right is None:
                        root.right = cls([])
                    cls.init_from_dict_extended({val: code[1:]}, root.right)
        return root

    def is_leaf(self):
        return self.left is None and self.right is None

    @property
    def max_leaf_depth(self) -> int:
        """Return the maximum leaf node depth of the binary tree.

        :return: Maximum leaf node depth.
        :rtype: int

        **Example**:

        .. doctest::

            >>> from binarytree import Node
            >>>
            >>> root = Node(1)
            >>> root.left = Node(2)
            >>> root.right = Node(3)
            >>> root.right.left = Node(4)
            >>> root.right.left.left = Node(5)
            >>>
            >>> print(root)
            <BLANKLINE>
              1____
             /     \\
            2       3
                   /
                  4
                 /
                5
            <BLANKLINE>
            >>> root.max_leaf_depth
            3
        """
        return _get_tree_properties(self).max_leaf_depth

    def level_order_encode(self) -> List[Tuple[Node, Tuple[int, ...]]]:
        """Return the nodes in the binary tree using level-order_ traversal.

        A level-order_ traversal visits nodes left to right, level by level.

        .. _level-order:
            https://en.wikipedia.org/wiki/Tree_traversal#Breadth-first_search

        :return: List of nodes.
        :rtype: [binarytree.Node]

        **Example**:

        .. doctest::

            >>> from binarytree import Node
            >>>
            >>> root = Node(1)
            >>> root.left = Node(2)
            >>> root.right = Node(3)
            >>> root.left.left = Node(4)
            >>> root.left.right = Node(5)
            >>>
            >>> print(root)
            <BLANKLINE>
                __1
               /   \\
              2     3
             / \\
            4   5
            <BLANKLINE>
            >>> root.levelorder
            [Node(1), Node(2), Node(3), Node(4), Node(5)]
        """

        current_nodes = [(self, ())]
        result = []

        while len(current_nodes) > 0:
            next_nodes = []
            for node, code in current_nodes:
                result.append((node, code))
                if node.left is not None:
                    next_nodes.append((node.left, code + (0,)))
                if node.right is not None:
                    next_nodes.append((node.right, code + (1,)))
            current_nodes = next_nodes

        return result

    @property
    def inorder(self) -> List[Node]:
        """Return the nodes in the binary tree using in-order_ traversal.

        An in-order_ traversal visits left subtree, root, then right subtree.

        .. _in-order: https://en.wikipedia.org/wiki/Tree_traversal

        :return: List of nodes.
        :rtype: [binarytree.Node]

        **Example**:

        .. doctest::

            >>> from binarytree import Node
            >>>
            >>> root = Node(1)
            >>> root.left = Node(2)
            >>> root.right = Node(3)
            >>> root.left.left = Node(4)
            >>> root.left.right = Node(5)
            >>>
            >>> print(root)
            <BLANKLINE>
                __1
               /   \\
              2     3
             / \\
            4   5
            <BLANKLINE>
            >>> root.inorder
            [Node(4), Node(2), Node(5), Node(1), Node(3)]
        """
        result: List[Node] = []
        stack: List[Node] = []
        node: Optional[Node] = self

        while node or stack:
            while node:
                stack.append(node)
                node = node.left
            if stack:
                node = stack.pop()
                result.append(node)
                node = node.right

        return result

    @property
    def leaves(self) -> List["Node"]:
        """Return the leaf nodes of the binary tree.

        A leaf node is any node that does not have child nodes.

        :return: List of leaf nodes.
        :rtype: [binarytree.Node]

        **Example**:

        .. doctest::

            >>> from binarytree import Node
            >>>
            >>> root = Node(1)
            >>> root.left = Node(2)
            >>> root.right = Node(3)
            >>> root.left.right = Node(4)
            >>>
            >>> print(root)
            <BLANKLINE>
              __1
             /   \\
            2     3
             \\
              4
            <BLANKLINE>
            >>> root.leaves
            [Node(3), Node(4)]
        """
        current_nodes = [self]
        leaves = []

        while len(current_nodes) > 0:
            next_nodes = []
            for node in current_nodes:
                if node.left is None and node.right is None:
                    leaves.append(node)
                    continue
                if node.left is not None:
                    next_nodes.append(node.left)
                if node.right is not None:
                    next_nodes.append(node.right)
            current_nodes = next_nodes
        return leaves

    def __repr__(self) -> str:
        """Return the string representation of the current node.

        :return: String representation.
        :rtype: str

        **Example**:

        .. doctest::

            >>> from utils.binary_tree import Node
            >>>
            >>> Node(1)
            Node(1)
        """
        return "Node({})".format(self.value)

    def __str__(self) -> str:
        """Return the pretty-print string for the binary tree.

        :return: Pretty-print string.
        :rtype: str

        **Example**:

        .. doctest::

            >>> from binarytree import Node
            >>>
            >>> root = Node(1)
            >>> root.left = Node(2)
            >>> root.right = Node(3)
            >>> root.left.right = Node(4)
            >>>
            >>> print(root)
            <BLANKLINE>
              __1
             /   \\
            2     3
             \\
              4
            <BLANKLINE>

        .. note::
            To include level-order_ indexes in the output string, use
            :func:`binarytree.Node.pprint` instead.

        .. _level-order:
            https://en.wikipedia.org/wiki/Tree_traversal#Breadth-first_search
        """
        lines = _build_tree_string(self, 0, False, "-")[0]
        return "\n" + "\n".join((line.rstrip() for line in lines))


if __name__ == '__main__':
    d = LABELS_MAP['FashionMNIST']
    for k, v in d.items():
        d[k] = v[1:]
    tree = Node.init_from_dict(d)
    print(tree)
