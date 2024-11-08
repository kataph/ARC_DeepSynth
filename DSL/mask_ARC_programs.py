import random
import re

HOLE = "<HOLE>"
HOLE_I = "HOLE"
HOLE_N = "<HOLE_>"
EXPANSION = "<EXP>"
EXPANSION_I = "EXP"

def parse_program(input_string):
    """
    Parses the program string into an AST (nested list structure).
    """
    tokens = re.findall(r'\(|\)|\w+|\S', input_string)
    stack = []
    current = []
    
    for token in tokens:
        if token == '(':
            new_node = []
            current.append(new_node)
            stack.append(current)
            current = new_node
        elif token == ')':
            if stack:
                current = stack.pop()
        else:
            current.append(token)
    
    return current[0]  # Return the parsed AST


def replace_random_nodes(ast, probability=0.3):
    """
    Recursively traverse the AST and randomly replace some nodes with <EXP>.
    Returns a modified copy of the AST with selected nodes replaced by <EXP>.
    """
    replaces: dict = {}
    
    def replace_nodes(node):
        # Decide if this node should be replaced
        if isinstance(node, list) and random.random() < probability:
            repl_index = len(replaces)
            replaces[repl_index]=node[0]

            return [HOLE_N.replace("_",str(repl_index))]
        
        # Otherwise, recursively replace in children
        return [replace_nodes(child) if isinstance(child, list) else child for child in node]
    
    return replace_nodes(ast), replaces


def ast_to_string(ast):
    """
    Convert the AST back into a string format similar to the input.
    """
    if isinstance(ast, list):
        return '( ' + ' '.join(ast_to_string(child) for child in ast) + ' )'
    else:
        return str(ast)

def produce_labels(output_string, replaces, return_only_one = False):
    if len(matches := list(re.finditer(HOLE_I+"([0-9]+)", output_string)))>0:
        random.shuffle(matches)
        ios=[]
        for to_expand in matches: 
            index = int(to_expand.groups()[0])
            label = replaces[index]
            start, end = to_expand.span()
            #label_string = output_string[:start-1] + replaces[index] + output_string[end+1:]
            feature_string = output_string[:start] + EXPANSION_I + output_string[end:]
            feature_string = re.sub("[0-9]+>", ">", feature_string)
            ios.append((feature_string, label))
            if return_only_one:
                break
        return ios
    else:
        return [(output_string, "")]
    
def mask_program(input_string, replace_prob=0.3, has_expansion = False, returns_only_one_masked_str = False):
    """if has_expansion returns list of io couples like [('( cover ( <EXP> ) )', 'merge_ct')]
    else it returns string with numbered random holes"""
    ast = parse_program(input_string)
    modified_ast, replaces = replace_random_nodes(ast, probability=replace_prob)
    output_string = ast_to_string(modified_ast)
    if has_expansion and not returns_only_one_masked_str:
        return produce_labels(output_string, replaces)
    if has_expansion and returns_only_one_masked_str:
        return produce_labels(output_string, replaces, return_only_one=True)[0][0] # only the feature strings with hole and exp tokens
    return output_string


if __name__ == "__main__":
    input_string = "( cover ( merge_ct ( vsplit ( var0 ) ( SEVEN ) ) ) ( identity ( connect ( RIGHT ) ( UNITY ) ) ) )"
    # input_string = "(underfill (pa0))"
    output = mask_program(input_string, replace_prob=0, has_expansion=True)
    print("Pruned AST with <EXP> tokens:")
    print(input_string)
    print(output)
