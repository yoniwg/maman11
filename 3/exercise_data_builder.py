import subprocess
import pyconll
import itertools 
import os.path
from util import get_single_file
from pathlib import Path

### data source transformer for first tasks

def build_verbs_data(source_treebank_name = "UD_Hebrew-HTB", git_hash = "82591c955e86222e32531336ff23e36c220b5846"):
    ''' transforms conllu to tokens with labels '''
    
    if not os.path.isdir(source_treebank_name) :
        print("fetching the data source from github...")
        subprocess.run(["git", "clone", "https://github.com/UniversalDependencies/" + source_treebank_name])
        subprocess.run(["git", "checkout", git_hash], cwd=source_treebank_name)
    
    print("transforming the data source...")
    conllu1 = pyconll.load_from_file(Path.cwd().joinpath(source_treebank_name, get_single_file(source_treebank_name, 'train')))
    conllu2 = pyconll.load_from_file(Path.cwd().joinpath(source_treebank_name, get_single_file(source_treebank_name, 'dev')))
    conllu3 = pyconll.load_from_file(Path.cwd().joinpath(source_treebank_name, get_single_file(source_treebank_name, 'test')))
    
    conllu = [conllu1, conllu2, conllu3]
        
    ## counting the occurences of words marked as verbs and non-verbs across our Hebrew Treebank

    verbs = {}
    non_verbs = {}

    skipped = 0
    for sentence in itertools.chain(*conllu):
        
        for token in sentence:
            
            # avoid including fused surface forms, 
            # only include the morphologically decomposed tokens 
            if not token.upos:
                skipped += 1
                continue 
            
            if token.upos == 'VERB':
                if token.form in verbs:
                    verbs.update({token.form : verbs[token.form]+1})
                else:
                    verbs.update({token.form : 0})
            else:
                if token.form in non_verbs:
                    non_verbs.update({token.form : non_verbs[token.form]+1})
                else:
                    non_verbs.update({token.form : 0})

    if skipped > 0: 
        print('skipped {:,} fused surface forms'.format(skipped))


    ## because many words have different part of speech roles, in the Hebrew Treebank, the same token 
    ## may appear sometimes as a verb and sometimes as a non-verb, so we put such tokens in a 3rd separate 
    ## class we will call for simplicity ― "dual"
                    
    dual = set(verbs.keys()) & set(non_verbs.keys()) # it's the set union operator here

    ## of course, the three sets need to be disjoint: ambiguous, verbs, non-verbs.
    ## the (right-hand side of the) next couple of lines are dict comprehensions ― 
    ## essentially that's just a one liner syntax for a loop

    verbs        = {k:v for (k,v) in verbs.items() if not k in dual} 
    non_verbs = {k:v for (k,v) in non_verbs.items() if not k in dual}

    return dict(verbs=verbs, non_verbs=non_verbs, dual=dual)



### data source transformer for last tasks

def _conllu_transformer(source):
    'transforms a conllu entry into an entry for our task'
    
    set = []
    
    for entry in itertools.chain(source): 
        
        sentence = []
        for idx, token in enumerate(entry):
            if '-' in token.id: 
                pass # python's no-op
            elif token.upos == 'VERB':
                sentence.append(dict(token=token.form, label=1))
            else:
                sentence.append(dict(token=token.form, label=0))
            
        set.append(sentence)    
    
    return set    


def build_verb_in_context_data():
    ''' transforms conllu to sentences with binary token labels '''
        
    if not os.path.isdir("UD_Hebrew-HTB") :
        print("fetching the data source from github...")
        subprocess.run(["git", "clone", "https://github.com/UniversalDependencies/UD_Hebrew-HTB"])
        subprocess.run(["git", "checkout", "82591c955e86222e32531336ff23e36c220b5846"], cwd="UD_Hebrew-HTB")

    print("transforming the data source...")
    
    train_set_source = pyconll.load_from_file('UD_Hebrew-HTB/he_htb-ud-train.conllu')
    test_set_source = pyconll.load_from_file('UD_Hebrew-HTB/he_htb-ud-dev.conllu')
            
    return _conllu_transformer(train_set_source), _conllu_transformer(test_set_source)
            
