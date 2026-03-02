import matplotlib.pyplot as plt

LEAF_DISTANCE = 1 / 2**11
DISTANCE_THRESHOLD = 0.002
MIN_X_DELTA = 1 / (2 ** (5))

def level_order_traversal(decision_tree):
    """
    Traverses a tree level order

    Args
        decision_tree

    Returns
        level order traversal of tree
    """
    queue = []
    queue.append(decision_tree)
    res = []
    while(len(queue) > 0):
        row = []
        for i in range(len(queue)):
            curr = queue[0]
            queue = queue[1:]
            row.append(curr)
            if not curr['leaf']:
                curr['left']['parent'] = curr
                curr['right']['parent'] = curr
                queue.append(curr['left'])
                queue.append(curr['right'])
        res.append(row)
    return res

def initialise_heights(decision_tree):
    """
    Initializes heights of tree in place

    Args
        decision tree
    """
    if decision_tree['leaf']:
        decision_tree['height'] = 1
    else:
        if not 'height' in decision_tree['left'].keys():
            initialise_heights(decision_tree['left'])
        if not 'height' in decision_tree['right'].keys():
            initialise_heights(decision_tree['right'])

        decision_tree['height'] = 1 + max(decision_tree['left']['height'], decision_tree['right']['height'])



def plot_node_level_order(decision_tree, depth, x, y, height, max_depth = 5):
    """
    Plots nodes by editing the positions of each row to prevent overlap

    Args
        decision_tree, depth, coordinates, height, max_depth
    
    Returns
        plots the tree
    """

    leaf_box = dict(boxstyle='round', facecolor='y')
    node_box = dict(boxstyle='round', facecolor='c')
    style = {"fontsize": 10, "ha": "center", "va": "center"}

    initialise_heights(decision_tree)

    level_order = level_order_traversal(decision_tree)

    for depth in range(len(level_order)):
        
        #initialicing coordinates of each node
        if depth > max_depth:
            return 
        
        if depth == 0:
            level_order[depth][0]['x'] = x 
            level_order[depth][0]['y'] = y 
        else:
            for node in level_order[depth]:
                node['x'] = 0
                node['y'] = node['parent']['y'] - height
        
        
        fix_row(level_order[depth], depth)
        

        #plotting all nodes in that row
        for k in range(len(level_order[depth])):
            node = level_order[depth][k]

            if node['leaf']:
                plt.text(node['x'], node['y'], '{}'.format(int(node['label'])), **style, bbox=leaf_box)
            else:
                plt.text(node['x'], node['y'], 'X{}<{}'.format(node["attribute"], node["value"]), **style, bbox=node_box)

            if 'parent' in node.keys():
                plt.plot([node['x'], node['parent']['x']], [node['y'], node['parent']['y']], color='m')
                plt.plot([node['x'], node['parent']['x']] , [node['y'], node['parent']['y']], color='m')
        
        

def fix_row(node_row, depth):
    """
    Positions nodes in each row according to their height, ensuring to remove overlaps.

    Args
        node_row, depth of node_row

    Returns
        this is done in place
    """

    total_height = sum(node['height'] for node in node_row)

    curr = 0

    #spacing the nodes in row according to their heights
    for node in node_row:
        node_breadth = node['height'] * MIN_X_DELTA 
        node['x'] = curr + 0.5 * MIN_X_DELTA * node['height'] / total_height
        curr += MIN_X_DELTA * node['height'] / total_height
    
    #ensuring leaves are evenly spaced
    i = 0
    while(i < len(node_row) - 1):
        if node_row[i]['leaf'] and node_row[i+1]['leaf']:
            node_row[i]['x'] = node_row[i]['parent']['x'] - LEAF_DISTANCE
            node_row[i + 1]['x'] = node_row[i+1]['parent']['x'] + LEAF_DISTANCE

        elif node_row[i]['leaf']:
            node_row[i]['x'] = node_row[i]['parent']['right']['x'] - LEAF_DISTANCE * 4
            
        
        elif node_row[i+1]['leaf']:
            node_row[i+1]['x'] = node_row[i+1]['parent']['left']['x'] + LEAF_DISTANCE * 4

        i+=2
    

    #ensuring left and right children are centred around parent
    k = 0 
    while(k < len(node_row) - 1):
        dist = node_row[k + 1]['x'] - node_row[k]['x'] 
        x_parent = node_row[k]['parent']['x']

        node_row[k]['x'] = x_parent - dist / 2
        node_row[k + 1]['x'] = x_parent + dist / 2

        k += 2

    #removing overlaps
    j = 1
    while(j < len(node_row) - 1):
        if abs(node_row[j]['x'] - node_row[j+1]['x']) < DISTANCE_THRESHOLD:

            dist_to_move = DISTANCE_THRESHOLD + node_row[j+1]['x'] - node_row[j]['x'] 
            node_row[j]['x'] -= dist_to_move  / 4
            node_row[j-1]['x'] -= dist_to_move  / 4
            node_row[j+1]['x'] += dist_to_move  / 4
            node_row[j+2]['x'] += dist_to_move  / 4
        
        if abs(node_row[j]['x'] - node_row[j+1]['x']) < DISTANCE_THRESHOLD:
            if not node_row[j]['leaf']:
                dist_to_move = DISTANCE_THRESHOLD + node_row[j+1]['x'] - node_row[j]['x'] 
                node_row[j]['x'] -= dist_to_move 
                node_row[j-1]['x'] -= dist_to_move 
            else:
                dist_to_move = DISTANCE_THRESHOLD + node_row[j+1]['x'] - node_row[j]['x']
                node_row[j+1]['x'] += dist_to_move 
                node_row[j+2]['x'] += dist_to_move 


        j+=2
  
    return
    

def plot_node(decision_tree, depth, x, y, height, max_depth=5):
    leaf_box = dict(boxstyle='round', facecolor='y')
    node_box = dict(boxstyle='round', facecolor='c')
    style = {"fontsize": 10, "ha": "center", "va": "center"}
    if depth > max_depth:
        # stop if depth is past max_depth
        return
    
    if decision_tree["leaf"]:
        plt.text(x, y, '{}'.format(int(decision_tree['label'])), **style, bbox=leaf_box)
    else:
        plt.text(x, y, 'X{}<{}'.format(decision_tree["attribute"], decision_tree["value"]), **style, bbox=node_box)
        new_y = y - height
        x_delta = max(1 / (2 ** (depth + 2)), MIN_X_DELTA)

        x_left, x_right = x - x_delta, x + x_delta

        # x_left, x_right = x - x_delta, x + x_delta
        plt.plot([x, x_left], [y, new_y], color='m')
        plot_node(decision_tree['left'], depth + 1, x_left, new_y, height, max_depth)
        plt.plot([x, x_right], [y, new_y], color='m')
        plot_node(decision_tree['right'], depth + 1, x_right, new_y, height, max_depth)


def visualise_tree(tree, name="test", max_depth=5):
    """
    Args:
        max_depth:
        tree (dict): the root of the tree

    creates a png image of the tree
    """
    initial_x = 0.5  # start halfway
    initial_y = 0  # start at top
    height = 1 / (max_depth + 0.2)
    plt.figure(figsize=(40, 10), dpi=100)

    plot_node_level_order(tree, 0, 0.5, 1, height, max_depth)
    plt.axis('off')
    plt.savefig("plots/"+name+".png")
    plt.show()


if __name__ == "__main__":
    tree = {"attribute": 1, "value": -10, "leaf": False,
            "left": {"attribute": 4, "value": 23, "leaf": False, "left": {"leaf": True, "label": 1},
                     "right": {"leaf": True, "label": 2}},
            "right": {"leaf": True, "label": 2}}
    visualise_tree(tree)