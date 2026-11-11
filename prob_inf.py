import argparse
import csv
from itertools import chain, permutations
import math
from random import random
import sys

DEBUG_OUTPUT=0

##############################################################################
## Exact inference

def calc_global_joint_prob(model, variableValues):
  """Return the joint probability of a complete Boolean assignment."""
  joint_prob = 1.0
  for var in model.vars:
    prob = read_cpt(model, var, variableValues)
    joint_prob *= prob if variableValues[var] else 1 - prob
  return joint_prob


def calc_query_exact_brute(model, queryVar, queryVal, evidence):
  """Compute P(queryVar=queryVal | evidence) by enumerating assignments."""
  query_and_evidence = 0.0
  evidence_prob = 0.0
  for assignment in truefalse_combination_iterator(model.vars):
    if all(assignment[var] == value for var, value in evidence.items()):
      prob = calc_global_joint_prob(model, assignment)
      evidence_prob += prob
      if assignment[queryVar] == queryVal:
        query_and_evidence += prob
  return query_and_evidence / evidence_prob


def calc_query_exact_tree(model, queryVar, queryVal, evidence):
  """Compute a posterior by recursively summing and multiplying factors."""
  calcOrder = generate_exact_inf_term_seq(model, queryVar, evidence)
  if DEBUG_OUTPUT > 0:
    print('Inf formula: ' + ' '.join(
      ('sum(' + v + ')') if marginalize else
      'P({0}|{1})'.format(v, ','.join(model.varDist[v].parents))
      for marginalize, v in calcOrder
    ))

  variableValues = {v: evidence.get(v) for v in model.vars}
  pr_true, pr_false = recurse_calc_query_exact_tree(
    model, queryVar, evidence, variableValues, calcOrder
  )
  total = pr_true + pr_false
  return (pr_true if queryVal else pr_false) / total


def recurse_calc_query_exact_tree(model, queryVar, evidence, variableValues, remainingCalc):
  """Evaluate the remaining factors for both values of the query variable.

  Summation terms branch over a hidden variable; probability terms multiply
  the corresponding CPT factor into the remainder of the calculation.
  """
  if DEBUG_OUTPUT > 0:
    indent = '    ' * (len(evidence) - sum(m for m, v in remainingCalc))

  marginalize, var = remainingCalc[0]
  if marginalize:
    if DEBUG_OUTPUT > 0:
      print(indent + 'Sum over ' + var)
    variableValues[var] = True
    true_T, true_F = recurse_calc_query_exact_tree(
      model, queryVar, evidence, variableValues, remainingCalc[1:]
    )
    variableValues[var] = False
    false_T, false_F = recurse_calc_query_exact_tree(
      model, queryVar, evidence, variableValues, remainingCalc[1:]
    )
    variableValues[var] = None
    return true_T + false_T, true_F + false_F

  if queryVar in model.varDist[var].parents:
    if DEBUG_OUTPUT > 0:
      print(indent + 'P({0}|{1}) [QC]'.format(var, ','.join(model.varDist[var].parents)))
    variableValues[queryVar] = True
    pr_true = read_cpt(model, var, variableValues)
    variableValues[queryVar] = False
    pr_false = read_cpt(model, var, variableValues)
    variableValues[queryVar] = None
    if not variableValues[var]:
      pr_true, pr_false = 1 - pr_true, 1 - pr_false
  elif var == queryVar:
    if DEBUG_OUTPUT > 0:
      print(indent + 'P({0}|{1}) [Q]'.format(var, ','.join(model.varDist[var].parents)))
    prob = read_cpt(model, var, variableValues)
    pr_true, pr_false = prob, 1 - prob
  else:
    if DEBUG_OUTPUT > 0:
      print(indent + 'P({0}|{1}) [S]'.format(var, ','.join(model.varDist[var].parents)))
    prob = read_cpt(model, var, variableValues)
    prob = prob if variableValues[var] else 1 - prob
    pr_true = pr_false = prob

  if len(remainingCalc) > 1:
    rest_true, rest_false = recurse_calc_query_exact_tree(
      model, queryVar, evidence, variableValues, remainingCalc[1:]
    )
    pr_true *= rest_true
    pr_false *= rest_false
  return pr_true, pr_false

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
    args.evidence={}
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