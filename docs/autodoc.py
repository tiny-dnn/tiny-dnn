# parse all layers and generate "layer catalogue" automatically

import glob
import re
import os

ROOT = 'https://github.com/tiny-dnn/tiny-dnn/blob/master/tiny_dnn/layers/'

# find class declarations
# returns: [posision, classname]
def get_layers(srcstring):
    layers = {}
    for m in re.finditer(r'class ([a-zA-Z_0-9]+_layer)', srcstring):
        layers[m.start(1)] = srcstring[m.start(1):m.end(1)]
    return layers

# find constructor declarations
# returns: array of string
def get_constructors(srcstring, layer):
    pat = r' +' + layer + r'\(.+?[^\(]\)'
    m = re.findall(pat, srcstring, re.M | re.DOTALL)
    return m

def replace_codeblock(comment):
    comment = re.sub(r'@code', '```cpp', comment)
    comment = re.sub(r'@endcode', '```', comment)
    return comment

def uncomment(comment):
    return re.sub(r'^[/\s]+?\*+/?', '', comment, flags=re.MULTILINE).strip()

def get_comment_before(srcstring, pos):
    cend = srcstring[:pos].rfind('*/')+2
    cbegin = srcstring[:pos].rfind('/*')

    m = re.search(r'\S+', srcstring[cend+1:pos-1])
    if m:
        return ''

    return replace_codeblock(uncomment(srcstring[cbegin:cend]))

# find c-style comment /* */ before the constructor
# returns: string
def get_constructor_comment(srcstring, constructor):
    pos = srcstring.find(constructor)
    if srcstring[:pos].rfind('explicit') >= (pos - len('explicit ')):
        pos = pos - len('explicit ') # adjustment for explicit parameter
    return get_comment_before(srcstring, pos)

def get_class_description(srcstring, beginofclass):

    cbegin = srcstring[:beginofclass-7].rfind('\n')  

    if cbegin > 0:
        # Special case for template class
        if re.match(r'template<.+>', srcstring[cbegin+1:]) is not None:
            return get_comment_before(srcstring, cbegin+1)

    return get_comment_before(srcstring, beginofclass-len('class '))

# returns: [arg, description] 
def get_parameter_descriptions(constructor_comments):
    params = {}

    for m in re.finditer('@param (\S+)( +\[in\]|\[out\]?)', constructor_comments):
        end = constructor_comments[m.start()+1:].find('@')

        if end == -1:
            dscr = constructor_comments[m.end(2)+1:]
        else:
            dscr = constructor_comments[m.end(2)+1:m.start()+end]

        dscr = dscr.lstrip().strip('* /\n\t')
        dscr = re.sub(r'\n[ *]+', '\n  ', dscr)
        params[m.group(1)] = dscr

    return params

def position_to_linenumber(content, pos):
    return content[:pos].count('\n') + 1;

def cpp_codeblock(code):
    return '```cpp\n' + code + '\n```\n';

def source_link(filename, content, pos):
    linenum = position_to_linenumber(content, pos)
    link = ROOT + os.path.basename(filename) + '#L' + str(linenum)
    return '[[source]](' + link + ')'

files= glob.glob('../tiny_dnn/layers/*.h')

docstring = ''

print '# Layers\n\n'

for file in files:
    f = open(file, 'r')
    srcstring = f.read()
    f.close()

    layers = get_layers(srcstring)

    for pos, layer in layers.items():
        blocks = []
        blocks.append('<span style="float:right;">' + 
                      source_link(file, srcstring, pos) +
                      '</span>')
        blocks.append('## ' + layer + '\n')

        class_description = uncomment(get_class_description(srcstring, pos))

        if class_description:
            blocks.append(class_description + '\n')

        ctors = get_constructors(srcstring, layer)

        blocks.append('### Constructors\n')

        for ctor in ctors:
            comment = get_constructor_comment(srcstring, ctor)
            if not comment:
                continue

            blocks.append(cpp_codeblock(ctor))

            descriptions = get_parameter_descriptions(comment)

            for k, v in descriptions.items():
                blocks.append("- **" + k + "** " + v + '\n')

        docstring += '\n'.join(blocks) + '\n'

print docstring
