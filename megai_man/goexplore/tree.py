from sys import getsizeof as size


class TreeNode:
    def __init__(self, parent=None, action=None):
        self.parent = parent
        self.action = action
        self.children = {}

    def __sizeof__(self):
        total = 0
        stack = [self.parent, self.action, self.children]
        while stack:
            object = stack.pop()
            if isinstance(object, TreeNode):
                stack.extend(
                    [id(object.parent), object.action, object.children]
                )
            elif isinstance(object, dict):
                for key, value in object.items():
                    stack.extend([key, value])
            else:
                total += size(object)
        return total

    def __del__(self):
        for action, child in self.children.items():
            del child

        del self.children
        del self.action

    def add(self, action):
        node = TreeNode(self, action)
        self.children[action] = node
        return node

    def has(self, action):
        return action in self.children

    def get(self, action):
        return self.children[action]

    def assign(self, code):
        self.code = code

    def remove(self):
        del self.code
        self.delete()

    def ascend(self):
        return self.parent


class LinkedTree:
    def __init__(self):
        self.root = TreeNode()
        self.node = self.root

    def __sizeof__(self):
        return size(self.root)

    def act(self, action):
        if self.node.has(action):
            self.node = self.node.get(action)
        else:
            self.node = self.node.add(action)

    def set(self, node):
        self.node = node

    def get_trajectory(self, temp=None):
        if temp is None:
            temp = self.node
        trajectory = []
        while not temp.action is None:
            trajectory.append(temp.action)
            temp = temp.parent
        return trajectory

    def size(self):
        count = 0
        stack = [self.root]
        while stack:
            node = stack.pop()
            count += len(node.children)
            stack.extend(list(node.children.values()))
        return count

    def add(self, trajectory, code):
        node = self.root
        for action in trajectory:
            node = node.add(action)
        node.assign(code)
        return node
