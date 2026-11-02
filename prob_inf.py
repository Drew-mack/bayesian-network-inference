import argparse
import csv
from itertools import chain, permutations
import math
#import matplotlib.pyplot as plt
#import numpy as np
from random import random
import sys

DEBUG_OUTPUT=0

##############################################################################
## Student code
def calc_global_joint_prob(model, variableValues):
  """
  Calculate the global joint probability of a model for a specific set of values
  
  model: model object, see read_model_file() for specification
  variableValues: dictionary of boolean values, keys are variable names
  """
  
  joint_prob = 1.0

  # multiply the probability of each variable
  for var in model.vars:
      prob = read_cpt(model, var, variableValues)

      # account for false variable assignments
      if not variableValues[var]:
          prob = 1 - prob

      joint_prob *= prob

  return joint_prob


def calc_query_exact_brute(model, queryVar, queryVal, evidence):
  """
  Calculate posterior probability for a given variable

  model: model object, see read_model_file() for specification
  queryVar: string, query variable name
  queryVal: boolean, value of the query variable we are calculating the probabilty for
  evidence: dictionary of boolean values, where keys are evidence variable names
            (Any variable not listed as query or evidence is assumed to be hidden)
  """

  # This first attempt at probabilistic inference will use the brute-force (table)
  #   enumeration approach shown in the Probability Intro slides (see slide 24)
  #
  # This requires the calculation of two joint probabilities based on the definition
  # of conditional probability:
  #                          Pr( Query & Evidence )
  #   Pr(Query | Evidence) = ----------------------
  #                              Pr( Evidence )
  #
  # Both of these joint probabilities can be calculated by going over every entry in
  # the global joint probability table and summing up the probabilities of those
  # entries that match what we're looking for
  
  #This is a helper function that should prove useful to you below
  def dict_issubset(d,sub):
    """
    Returns True if every key,value pair in sub has a matching key and value in d
    Keys in d that _are not_ in sub are ignored.
    Note: sub should not contain any entries with value None
    """
    return all(d.get(key,None)==val for key,val in sub.items())
     
  pr_QE=0 #Accumulate probabilities for Query and Evidence
  pr_E=0  #Accumulate probabilities for just Evidence
  #Generate table by looping over every possible combination of variables
  for jptEntry in truefalse_combination_iterator(model.vars):
    #Get joint probability for this combination of variables
    pr_entry=calc_global_joint_prob(model,jptEntry)

    
    # sum probabilities matching the evidence
    if dict_issubset(jptEntry, evidence):
        pr_E += pr_entry

        # also check whether the query matches
        if jptEntry[queryVar] == queryVal:
            pr_QE += pr_entry


  #Definition of coditional probability
  return pr_QE/pr_E

def calc_query_exact_tree(model, queryVar, queryVal, evidence):
  """
  Calculate posterior probability for a given variable

  model: model object, see read_model_file() for specification
  queryVar: string, query variable name
  queryVal: boolean, value of the query variable we are calculating the probabilty for
  evidence: dictionary of boolean values, where keys are evidence variable names
            (Any variable not listed as query or evidence is assumed to be hidden)
  """
  
  # First step, we need to figure out what order we will calculate terms in and where
  # marginalization needs to happen.
  #
  # That said, though this is a part of the inference process that you need to know, it's a
  # bit tricky to get working in general, especially the optimization bits.
  #
  # So I have provided an implementation for this below. If you're curious, feel free to have a look.
  calcOrder=generate_exact_inf_term_seq(model,queryVar,evidence)
  # This will return a list of (boolean,string) tuples that indicates which parts need to be calculated in which order.
  # True indicates a summation (i.e. marginalization) term, False indicates a probability term.
  # For example, the formula on slide 20 would be represented as:
  # [ (True,'A'), (True,'E'), (False,'J'), (False,'M'), (False,'A'), (False,'B'), (False,'E') ]
  # The formula on slide 21 would be:
  # [ (False,'B'), (True,'A'), (False,'J'), (False,'M'), (True,'E'), (False,'A'), (False,'E') ]
  # Some marginalization terms for hidden variables, and probability terms for any variables, may be missing
  # if my code determines they can be optimized away (e.g., handled by normalization instead).
  
  # Debug: Output a nicer version of the calculation order (inference formula)
  if DEBUG_OUTPUT>0: print('Inf formula: '+' '.join( ( ('sum('+v+')') if m else 'P({0}|{1})'.format(v,','.join(model.varDist[v].parents)) ) for m,v in calcOrder))
  
  #Make a dictionary with entries for every possible variable, and their values where available (None otherwise)
  variableValues={v:evidence.get(v,None) for v in model.vars}
  
  # Next step, implement the calculation
  #
  # I strongly recommend using a recursive solution, in which case leave the below line of code
  # and move on to implement the recurse_calc_query_exact_tree() function
  prQ_T,prQ_F=recurse_calc_query_exact_tree(model,queryVar,evidence,variableValues,calcOrder)
  # HOWEVER, you are not required to implement recursively, in which case delete the above line
  # and associated function and add your own calculation code here
  
  # normalize the query probabilities
  total = prQ_T + prQ_F

  if queryVal:
    return prQ_T / total
  else:
    return prQ_F / total


def recurse_calc_query_exact_tree(model, queryVar, evidence, variableValues, remainingCalc):
  """
  Recursiving process the summation tree 
  
  model,queryVar,evidence: See calc_query_exact_tree()
  variableValues: dictionary of boolean values or None, values for entire set of variables or None if no value set yet
    Note: You MAY change this structure during the recursion, but make sure undo those changes when you're done with them
  remainingCalc: list of (boolean,string), see XXX and comments in calc_query_exact_tree() for format
  """
  if DEBUG_OUTPUT>0: indent='    '*(len(evidence)-sum(m for m,v in remainingCalc)) #Indent based on how deep in the recursion we are

  # Your overall task in the function is to assign values to:
  #   prQ_T
  #   prQ_F
  # Which should (eventually) contain the (relative) probabilities for the remainder of the calculation
  # covering both cases prQ_T (where Query=True) and prQ_F (where Query=False).
  #
  # WARNING: In order to keep the code simpler, we will always use always use both
  # prQ_T and prQ_F. In some cases that means the same value will be assigned to both.

  marginalize,var=remainingCalc[0] #Grab details for the next term we have to deal with
  # 'var' above will be very important for you below, this is the current
  # model variable we are calculating for
  # (You can ignore 'marginalize', only my code needs to use that one.)
  if marginalize:
    #Summation term, need to branch over all possible values and continue calculation
    if DEBUG_OUTPUT>0: print(indent+'Sum over '+var)
    
    # calculate both possible values of the hidden variable
    variableValues[var] = True
    true_T, true_F = recurse_calc_query_exact_tree(
      model, queryVar, evidence, variableValues, remainingCalc[1:]
    )

    variableValues[var] = False
    false_T, false_F = recurse_calc_query_exact_tree(
      model, queryVar, evidence, variableValues, remainingCalc[1:]
    )

    # restore the hidden variable
    variableValues[var] = None

    # sum the probabilities from both branches
    prQ_T = true_T + false_T
    prQ_F = true_F + false_F

  else:
    #Probability term, calculate conditional probability for this variable and continue calculation
    prQ_T, prQ_F = 1,1 #Base case if we don't recurse below
    if queryVar in model.varDist[var].parents:
      #Query variable is a condition for this term
      if DEBUG_OUTPUT>0: print(indent+'P({0}|{1}) [QC]'.format(var,','.join(model.varDist[var].parents)))

      # calculate probability when query is true
      variableValues[queryVar] = True
      prQ_T = read_cpt(model, var, variableValues)

      # calculate probability when query is false
      variableValues[queryVar] = False
      prQ_F = read_cpt(model, var, variableValues)

      # restore the query variable
      variableValues[queryVar] = None

      # account for false variable assignments
      if not variableValues[var]:
        prQ_T = 1 - prQ_T
        prQ_F = 1 - prQ_F

    elif var==queryVar:
      #This term is probability _for_ the Query variable
      if DEBUG_OUTPUT>0: print(indent+'P({0}|{1}) [Q]'.format(var,','.join(model.varDist[var].parents)))

      # calculate probability for both query values
      prob = read_cpt(model, var, variableValues)

      prQ_T = prob
      prQ_F = 1 - prob

    else:
      #Simple term, no need to worry about query variable
      if DEBUG_OUTPUT>0: print(indent+'P({0}|{1}) [S]'.format(var,','.join(model.varDist[var].parents)))

      
      # calculate the probability for this variable
      prob = read_cpt(model, var, variableValues)

      # account for false assignments
      if not variableValues[var]:
        prob = 1 - prob

      prQ_T = prob
      prQ_F = prob


    if len(remainingCalc)>1:
      #If there are still terms left, then recurse
      recurse_prQ_T, recurse_prQ_F = recurse_calc_query_exact_tree(model,queryVar,evidence,variableValues,remainingCalc[1:])
      
      # multiply current probabilities by remaining factors
      prQ_T *= recurse_prQ_T
      prQ_F *= recurse_prQ_F

  return prQ_T, prQ_F #Return (relative) probability that query is True vs False
  
##############################################################################
## Support code
def read_cpt(model,varName,condVals):
  """
  Read conditional probability for a specified variable with provided condition (parent) values
  Note, the value returned is conditional probability for variable being True
  
  Hint: This function is engineered so that condVals can be a dictionary containing
  more entries than necessary, so that you do not have to extract entries out of
  another dictionary to use this function.
  
  model: model object, see read_model_file() for specification
  varName: string, variable name to read probability for
  condValues: dictionary of boolean values, where keys are condition/parent names for the specificed variable
              (Missing conditions will cause errors, extraneous values will be ignored)
  """
  if varName not in model.varDist:
    raise ValueError("Variable '{0}' not in model".format(varName))
  varDist=model.varDist[varName]
  key=frozenset(((x,condVals.get(x,None)) for x in varDist.parents))
  if key not in varDist.cpt:
    if all(condVals.get(x,None) is not None for x in varDist.parents):
      raise IndexError("CPT for variable '{0}' has no entry matching:\n{1}".format(varName,"\n".join("{0}={1}".format(x,v) for x,v in key)))
    else:
      raise IndexError("condVals lacks values for some parents of variable '{0}':\n{1}".format(varName,"\n".join(x for x,v in key if v is None)))
  return varDist.cpt[key]

def truefalse_combination_iterator(entries):
  """
  Create a sequence of dictonaries contain all possible combinations of True and False for each entry in 'entries'
  """
  entries=list(entries)
  entries.reverse()
  if len(entries)>30:
    error('truefalse_combination_iterator() does not support more than 30 entries at this time')
  for c in range(1<<len(entries)):
    yield {x:(c&(1<<i))>0 for x,i in zip(entries,range(len(entries)))}

def generate_exact_inf_term_seq(model,queryVar,evidence):
  """
  Create represention of terms in an inference calculation such as on slides 20-21
  
  Returns a list of (boolean,string) tuples where:
    (True,variable) represents a summation term where a variable needs to be marginalized
    (False,variable) represents a probability term where the conditional probability of a term needs to be included
  """
  hiddenVars=tuple(v for v in model.vars if (v!=queryVar and v not in evidence))

  #--------------------------------------------------------
  # Naive solution
  #
  # model.varsDep already has variables in order of dependency...
  # So take that and insert summation terms any time we encounter a new hidden variable
  #
  # Downside is little optimization, likely to have many unnecessary terms
  if False:
    hiddenLeft=set(hiddenVars)
    seq=[]
    for v in model.varsDep:
      #Check if factor variable is a (unhandled) hidden variable
      if v in hiddenLeft:
        seq.append( (True,v) ) #If so, trigger a marginalization
        hiddenLeft.remove(v)   #And mark it as handled
      for p in model.varDist[v].parents:
        #Check if a condition is a (unhandled) hidden variable, etc etc
        if p in hiddenLeft:
          seq.append( (True,p) )
          hiddenLeft.remove(p)
      seq.append( (False,v) ) #Then process the factor itself
    assert(len(hiddenLeft)==0)

  #--------------------------------------------------------
  # Arbitrary ordering
  #
  # What if we wanted to handle hidden variables in an arbitrary order?
  #
  # Possible, but we'll have to be careful where we put factors, after
  # all their dependencies are satisfied.
  def seq_from_hid_order(hOrd):
    #The trick to make this work is to first assign every
    #hidden variable a priority based on the order
    prio={h:i for i,h in enumerate(hOrd)}
    prio.update((v,-1) for v in model.vars if v not in prio) #non-hidden variables get lowest prio so they don't count
    #Then rate each factor on the highest priority amongst its dependencies
    vOrd=[(max(chain((prio[v],),(prio[c] for c in model.varDist[v].parents))),True,v) for v in model.vars]
    vOrd.extend( (prio[h],False,h) for h in hOrd ) #Add placeholers for summations as well, the False ensures these will sort before their dependents
    vOrd.sort()
    #All that's left is to turn it into the expected sequence format
    return list( (not nm,v) for _,nm,v in vOrd )
  
  #--------------------------------------------------------
  # Brute force best
  #
  # Now, where to get an ordering to use the above?
  #
  # We could brute force try every possible ordering...
  if True:
    bestSeq=None
    bestSeqCost=sys.maxsize
    for hOrd in permutations(hiddenVars):
      tSeq=seq_from_hid_order(hOrd)
      #Note, really should do below norm optimization here too
      
      #Now the tricky bit is to rate each ordering
      #We'll do it by doubling the cost of each factor every time
      #We cross a summation
      tot=0
      ct=1
      for m,v in tSeq:
        if m:
          ct*=2
        else:
          tot+=ct
      
      if tot<bestSeqCost:
        bestSeq=tSeq
        bestSeqCost=tot
    seq=bestSeq
  # But this will be very expensive for large models
  #--------------------------------------------------------
  # Greedy
  #
  # Alternately, we could apply a greedy approach.
  #
  # Some how rate each hidden variable on how expensive we think
  # it is, then put the most expensive ones earliest
  # ***TODO***

  #--------------------------------------------------------
  # Simple normalization optimization
  #
  # One thing we learned is that for a multiplicative term,
  # if it doesn't mention the query variable, then it's a
  # constant and can be handled via normalization (folded into alpha)
  #
  # This is non-trivial to detect for summation terms, but we
  # can easily do it for factors outside of any summation...
  if True:
    i=0
    while i<len(seq):
      m,v=seq[i]
      if m:
        break #Found first summation, quit
      if v!=queryVar and all(cv!=queryVar for cv in model.varDist[v].parents):
        #No mention of query variable, remove
        del seq[i]
      else:
        i+=1
  
  return seq

def calc_query_approx(model,queryVar,queryVal,evidence):
  raise NotImplementedError()

def generate_joint_prob_table(model):
  """
  Output a joint probability table for the provided model
  """
  from tabulate import tabulate
  table=[]
  row=list(model.vars)
  row.append('Joint Pr')
  table.append(row)
  for varVals in truefalse_combination_iterator(model.vars):
    pr=calc_global_joint_prob(model,varVals)
    row=[varVals[x] for x in model.vars]
    row.append(pr)
    table.append(row)
  print(tabulate(table, headers='firstrow', tablefmt='fancy_grid'))
  return

def read_model_file(filename):
  """
  Returns model object with the following elements:
    vars : list of strings
      The list of variables the model describes
      In alphabetical order
    varsDep : list of strings
      Same contents as 'vars' but in dependency order (parents come before children)
    varDist : dict of objects
      Distribution information for each variable
      Dictionary key is variable name
      Object has the following elements:
        parents : set of strings
        children : set of strings
        cpt : dict of numbers
          Conditional probability table for variable, i.e., probability of variable true given each combination of parent values
          Dictionary key is a set of (var_name,var_value) tuples containing values for all parents (and nothing else)
            From dict: cpt[frozenset(((x,dict[x]) if x in dict else (x,None) for x in parents))]
  Model file format is as follows:
    Basic file format is Comma-Separated Value (.csv)
    File contains multiple tables, one table per variable representing that variable's conditional probability table
    Tables are separated by atleast one empty line
    Any row that starts with '#' (excluding whitespace) will be treated as a comment and skipped
    Each table:
      Starts with a header row containing variable names
        The last name is the variable whose cond probability is being described
        Any preceding names are considered to be parent variables
      Following rows contain True/False values for each parent and probability for main variable being true
      Any missing parent value combinations will be assumed to be probability 0.5
    Only Bernoulli/Boolean variables can be represented in this file format
  """
  class ModelObj:
    def __init__(self):
      self.vars=[]
      self.varsDep=None
      self.varDist={}

  class VarObj:
    def __init__(self, parents, children, cpt):
      self.parents = parents
      self.children = children
      self.cpt = cpt

  model=ModelObj()
  #--------------------------------------------------------
  #Read data from file
  with open(filename, newline='', encoding='utf_8_sig') as csvfile:
    csvreader = csv.reader(csvfile)
    
    rowNum=0
    var=None
    varIdx=None
    parents=None
    cpt=None
    for row in ([e for e in x if len(e)>0] for x in chain(csvreader,[[]])):
      rowNum+=1
      srow=''.join(row).strip()
      if srow.startswith('#'):
        continue #Comment line, skip
      if len(srow)==0:
        #Empty line
        if var is not None:
          #End current table
          model.vars.append(var)
          model.varDist[var]=VarObj(frozenset(parents),None,cpt)
          #Wait for new table
          var=None
          varIdx=None
          parents=None
          cpt=None
      elif var is None:
        #Start new table
        if len(row)>1:
          parents=row[0:-1]
        else:
          parents=[]
        varIdx=len(row)-1
        var=row[varIdx]
        cpt={}
        print("Reading table for variable '{0}'...".format(var)) 
      else:
        #Add new entry to table
        if len(row)<varIdx+1:
          error("Malformat in csv line {0}: Too few columns for parent values and variable probability".format(rowNum))
        if len(parents)>0:
          key=frozenset(zip(parents,(e.strip().upper().startswith('T') for e in row[0:-1])))
        else:
          key=frozenset()
        cpt[key]=float(row[-1])
  model.vars.sort()
  #--------------------------------------------------------
  # Check distributions for missing entries
  vCheck=frozenset(model.vars)
  for var in model.vars: #Make sure every mentioned variable has an entry
    for p in model.varDist[var].parents:
      if p not in vCheck:
        error("Variable '{0}' has '{1}' as parent, but variable '{1}' was not defined".format(var,p))
  for var in model.vars: #Check every cpt for missing rows
    varDist=model.varDist[var]
    missingCnt=0
    for varVals in truefalse_combination_iterator(varDist.parents):
      key=frozenset(((x,v) for x,v in varVals.items()))
      if key not in varDist.cpt:
        missingCnt+=1
        varDist.cpt[key]=0.5
    if missingCnt>0:
      print("Warning: read_model_file(): Variable '{0}' had {1} missing entries, filled with 0.5".format(var,missingCnt))
  #--------------------------------------------------------
  # Create children entries
  for var in model.vars:
    model.varDist[var].children=set()
  for var in model.vars:
    for p in model.varDist[var].parents:
      model.varDist[p].children.add(var)
  for var in model.vars:
    model.varDist[var].children=frozenset(model.varDist[var].children)
  #--------------------------------------------------------
  # Create dependency ordering
  varsDep=[x for x in model.vars if len(model.varDist[x].parents)==0] #Start from prior variables (no parents)
  idx=0
  parentsLeft={x:len(model.varDist[x].parents) for x in model.vars} #Track how many of a node's parents are still not in the ordering
  while idx<len(varsDep):
    var=varsDep[idx]
    for c in model.varDist[var].children:
      parentsLeft[c]-=1
      if parentsLeft[c]==0:
        #All parents have been visited, so dependencies of this child are met
        varsDep.append(c)
      elif parentsLeft[c]<0:
        #Repeat visit to a parent can only happen if a cycle exists
        error("Cycle in graph detected, involving variable '{0}'".format(var))
    idx+=1
  model.varsDep=varsDep
  #--------------------------------------------------------
  return model

def print_model(model):
  """
  Print a model object back out in pretty form
  """
  from tabulate import tabulate
  for v in model.vars:
    varDist=model.varDist[v]
    print('--------------------------------------------------')
    print('Variable:',v)
    print('--------------------------------------------------')
    print('Children:',', '.join(varDist.children))
    
    table=[]
    row=list(varDist.parents)
    row.append('P({0}=T|...)'.format(v))
    table.append(row)
    for varVals in truefalse_combination_iterator(varDist.parents):
      pr=read_cpt(model,v,varVals)
      row=[varVals[x] for x in varDist.parents]
      row.append(pr)
      table.append(row)
    print(tabulate(table, headers='firstrow', tablefmt='fancy_grid'))
    print("")
    
##############################################################################
## Main functions
def main(args):
  global DEBUG_OUTPUT
  if args.debug:
    DEBUG_OUTPUT=1
  #Argument checking plus additional parsing
  if args.mode=='table' and ( args.query is not None or args.evidence is not None ):
    error('Arguments --query and --evidence not allowed in table mode')
  if args.mode=='print' and ( args.query is not None or args.evidence is not None ):
    error('Arguments --query and --evidence not allowed in print mode')
  if args.mode!='table' and args.mode!='print' and ( args.query is None ):
    error('Argument --query required in inference modes')
  elif args.query is not None:
    if '=' not in args.query:
      error('Query variable malformed, must follow VariableName=True or VariableName=False format')
    s=args.query.split('=')
    args.query=(s[0].strip(),s[1].strip().upper().startswith('T'))
  if args.evidence is None:
    args.evidence=[]
  else:
    ev=[]
    for e in args.evidence:
      if '=' not in e:
        error("Evidence argument '{0}' malformed, must follow VariableName=True or VariableName=False format".format(e))
      s=e.split('=')
      ev.append( (s[0].strip(),s[1].strip().upper().startswith('T')) )
    args.evidence={ var:val for var,val in ev }

  print('Reading model from',args.model)
  model=read_model_file(args.model)

  if args.mode=='table':
    generate_joint_prob_table(model)
  elif args.mode=='print':
    print_model(model)
  else:
    #One of the inference modes
    #Check inputs against model
    if args.query[0] not in model.vars:
      error("'{0}' is not a variable in supplied model".format(args.query[0]))
    for var,val in args.evidence.items():
      if var not in model.vars:
        error("'{0}' is not a variable in supplied model".format(var))
    #Output problem setup
    print("Inference mode:",args.mode)
    print("Query: '{0}' is {1}".format(args.query[0],args.query[1]))
    if len(args.evidence)==0:
      print("No evidence")
    else:
      print("Evidence:")
      for var,val in args.evidence.items():
        print("  '{0}' is {1}".format(var,val))

    #Run inference
    pr=None
    if args.mode=='brute':
      pr=calc_query_exact_brute(model,args.query[0],args.query[1],args.evidence)
    elif args.mode=='tree':
      pr=calc_query_exact_tree(model,args.query[0],args.query[1],args.evidence)
    else: #args.mode=='approx'
      pr=calc_query_approx(model,args.query[0],args.query[1],args.evidence)
    print('Probability is',pr)

  return

def error(msg):
  print(msg)
  sys.exit(1)
  return

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description="CSE3521 Homework 3 - Probabilistic Inference")
  parser.add_argument('--model', type=str, action='store', required=True, help='Input file to load model from')
  parser.add_argument('--mode', type=str, action='store', choices=['brute', 'tree', 'approx', 'table', 'print' ], required=True, help='How to process the model')
  parser.add_argument('--query', '-q', type=str, action='store', help='Query variable to perform inference on, in format VariableName=True or VariableName=False')
  parser.add_argument('--evidence', '-e', type=str, action='append', help='Evidence variable and value, in format VariableName=True or VariableName=False\nRepeat argument for multiple variables')
  parser.add_argument('--debug', action='store_true', default=False, help='Enable debugging output statements')
  args = parser.parse_args()
  error=lambda msg : parser.error(msg)
  main(args)