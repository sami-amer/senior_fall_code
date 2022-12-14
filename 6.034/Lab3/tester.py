#!/usr/bin/env python3

# MIT 6.034 Constraints Lab

import xmlrpc.client
import traceback
import sys
import os
import tarfile
from io import BytesIO
from constraint_api import *
from test_problems import constraint_or

python_version = sys.version_info
is_windows = sys.platform in ["win32", "cygwin"]
if python_version.major != 3:
    raise Exception("Illegal version of Python for 6.034 lab. Detected Python "
                    + "version is: " + str(sys.version))
if python_version.minor == 5 and python_version.micro <= 1:
    raise Exception("Illegal version of Python; versions 3.5.0 and 3.5.1 are disallowed "
                    + "due to bugs in their XMLRPC libraries. Detected version is: "
                    + str(sys.version))


def test_summary(dispindex, ntests):
    return "Test %d/%d" % (dispindex, ntests)

def show_result(testsummary, testcode, correct, got, expected, verbosity):
    """ Pretty-print test results """
    if correct:
        if verbosity > 0:
            print("%s: Correct." % testsummary)
        if verbosity > 1:
            print_testcode(testcode)
            print()
    else:
        print("%s: Incorrect." % testsummary)
        print_testcode(testcode)
        print("Got:     ", got, "\n")
        print("Expected:", expected, "\n")

def print_testcode(testcode):
    if isinstance(testcode, (tuple, list)) and len(testcode) >= 3:
        print('\t', testcode[2])
    else:
        print('\t', testcode)

def show_exception(testsummary, testcode):
    """ Pretty-print exceptions (including tracebacks) """
    print("%s: Error." % testsummary)
    print("While running the following test case:")
    print_testcode(testcode)
    print("Your code encountered the following error:")
    traceback.print_exc()
    print()


def get_lab_module(online=False):
    lab = __import__('lab_constraints')
    return lab

# encode/decode Constraint and ConstraintSatisfactionProblem objects
def constraint_greater_than(a,b):
    return a > b
constraint_dict = {'constraint_equal': constraint_equal,
                   'constraint_different': constraint_different,
                   'constraint_or': constraint_or,
                   'constraint_greater_than': constraint_greater_than}
def encode_constraint(constraint):
    fn_name = constraint.constraint_fn.__name__
    if fn_name == '<lambda>':
        print((' ** Note: Unfortunately, the online tester is unable to accept '
               +'lambda functions. To pass the online tests, please use a '
               +'pre-defined named function instead. **'))
    elif fn_name not in constraint_dict:
        print(('Error: Constraint function ' + fn_name + ' cannot be transmitted '
               +'to server.  Please use a pre-defined constraint function instead.'))
    return [constraint.var1, constraint.var2, fn_name]
def decode_constraint(var1, var2, constraint_fn_name):
    return Constraint(var1, var2, constraint_dict[constraint_fn_name])

def encode_CSP(csp):
    return [csp.variables, list(map(encode_constraint, csp.constraints)),
            csp.unassigned_vars, csp.domains, csp.assignments]
def decode_CSP(variables, constraint_list, unassigned_vars, domains, assignments):
    csp = ConstraintSatisfactionProblem(variables)
    csp.constraints = [decode_constraint(*c_args) for c_args in constraint_list]
    csp.unassigned_vars = unassigned_vars
    csp.domains = domains
    csp.assignments = assignments
    return csp

# decode functions received from server
def lambda_F(p, v): return False
def lambda_T(p, v): return True
def lambda_1(p, v): return len(p.get_domain(v))==1
def lambda_12(p, v): return len(p.get_domain(v)) in [1, 2]
def lambda_B(p, v): return v=='B'
def lambda_BC(p, v): return v in 'BC'
function_dict = {'lambda_F': lambda_F, 'lambda_T': lambda_T,
                 'lambda_1': lambda_1, 'lambda_B': lambda_B,
                 'lambda_12': lambda_12, 'lambda_BC': lambda_BC}


def type_decode(arg, lab):
    """
    XMLRPC can only pass a very limited collection of types.
    Frequently, we want to pass a subclass of 'list' in as a test argument.
    We do that by converting the sub-type into a regular list of the form:
    [ 'TYPE', (data) ] (ie., AND(['x','y','z']) becomes ['AND','x','y','z']).
    This function assumes that TYPE is a valid attr of 'lab' and that TYPE's
    constructor takes a list as an argument; it uses that to reconstruct the
    original data type.
    """
    if isinstance(arg, list) and len(arg) >= 1: # There is no future magic for tuples.
#        if arg[0] == 'Constraint': # not used because no lone Constraints (ie outside of a CSP) are sent from server
#            return decode_constraint(*type_decode(arg[1], lab))
        if arg[0] == 'CSP':
            return decode_CSP(*type_decode(arg[1], lab))
        elif arg[0] == 'callable':
            return function_dict[arg[1]]
        else:
            try:
                mytype = arg[0]
                data = arg[1:]
                return getattr(lab, mytype)([ type_decode(x, lab) for x in data ])
            except AttributeError:
                return [ type_decode(x, lab) for x in arg ]
            except TypeError:
                return [ type_decode(x, lab) for x in arg ]
    else:
        return arg

def is_list_of_constraints(arg):
    return (arg != [] and isinstance(arg, (tuple, list))
            and all(map(isinstance_Constraint, arg)))

def type_encode(arg):
    "Encode objects as lists in a way that can be decoded by 'type_decode'"
    if isinstance_Constraint(arg):
        return [ 'Constraint', type_encode(encode_constraint(arg)) ]
    elif (isinstance(arg, list) and len(arg) == 2 # special case for FUNCTION_WITH_CSP
          and isinstance_ConstraintSatisfactionProblem(arg[1])):
        return [type_encode(arg[0]), type_encode(encode_CSP(arg[1]))]
    elif is_list_of_constraints(arg): # special case for all_different
        return ['list-of-constraints', list(map(encode_constraint, arg))]
    else:
        return arg

def run_test(test, lab):
    """
    Takes a 'test' tuple as provided by the online tester
    (or generated by the offline tester) and executes that test,
    returning whatever output is expected (the variable that's being
    queried, the output of the function being called, etc)

    'lab' (the argument) is the module containing the lab code.

    'test' tuples are in the following format:
      'id': A unique integer identifying the test
      'type': One of 'VALUE', 'FUNCTION', 'MULTIFUNCTION', or 'FUNCTION_ENCODED_ARGS'
      'attr_name': The name of the attribute in the 'lab' module
      'args': a list of the arguments to be passed to the function; [] if no args.
      For 'MULTIFUNCTION's, a list of lists of arguments to be passed in
    """
    id, mytype, attr_name, args = test

    attr = getattr(lab, attr_name)

    if mytype == 'VALUE':
        return attr
    elif mytype == 'FUNCTION':
        return attr(*args)
    elif mytype == 'FUNCTION_WITH_CSP':
        # return modified version of input csp
        for a in args:
            if isinstance_ConstraintSatisfactionProblem(a):
                return [attr(*args), a]
        raise Exception("Test Error: 'FUNCTION_WITH_CSP' test missing CSP. "
                        + "Please contact a TA if you see this error.")
    elif mytype == 'MULTIFUNCTION':
        return [ run_test( (id, 'FUNCTION', attr_name, FN), lab)
                for FN in type_decode(args, lab) ]
    elif mytype == 'FUNCTION_ENCODED_ARGS':
        return run_test( (id, 'FUNCTION', attr_name, type_decode(args, lab)), lab )
    elif mytype == 'FUNCTION_ENCODED_ARGS_WITH_CSP':
        return run_test( (id, 'FUNCTION_WITH_CSP', attr_name, type_decode(args, lab)), lab )
    else:
        raise Exception("Test Error: Unknown TYPE: " + str(mytype)
                        + ".  Please make sure you have downloaded the latest"
                        + "version of the tester script.  If you continue to "
                        + "see this error, contact a TA.")

def test_offline(verbosity=1):
    """ Run the unit tests in 'tests.py' """
    import tests as tests_module

    tests = tests_module.get_tests()
    ntests = len(tests)
    ncorrect = 0

    for index, (testname, getargs, testanswer, expected, fn_name, type) in enumerate(tests):
        dispindex = index+1
        summary = test_summary(dispindex, ntests)

        try:
            if callable(getargs):
                getargs = getargs()

            answer = run_test((index, type, fn_name, getargs), get_lab_module())
        except NotImplementedError:
            print("%d: (%s: Function not yet implemented, NotImplementedError raised)" % (dispindex, testname))
            continue
        except Exception:
            show_exception(summary, testname)
            continue

        # This prevents testanswer from throwing errors. eg, if return type is
        # incorrect, testanswer returns False instead of raising an exception.
        try:
            correct = testanswer(answer)
        except NotImplementedError:
            print("%d: (%s: No answer given, NotImplementedError raised)" % (dispindex, testname))
            continue
        except (KeyboardInterrupt, SystemExit): # Allow user to interrupt tester
            raise
        except Exception:
            correct = False

        show_result(summary, testname, correct, answer, expected, verbosity)
        if correct: ncorrect += 1

    print("Passed %d of %d tests." % (ncorrect, ntests))
    return ncorrect == ntests

def get_target_upload_filedir():
    """ Get, via user prompting, the directory containing the current lab """
    cwd = os.getcwd() # Get current directory.  Play nice with Unicode pathnames, just in case.

    print("Please specify the directory containing your lab,")
    print("or press Enter to use the default directory.")
    print("Note that all files from this directory will be uploaded!")
    print("Labs should not contain large amounts of data; very large")
    print("files will fail to upload.")
    print()
    print("The default path is '%s'" % cwd)
    target_dir = input("[%s] >>> " % cwd)

    target_dir = target_dir.strip()
    if target_dir == '':
        target_dir = cwd

    print("Ok, using '%s'." % target_dir)

    return target_dir

def get_tarball_data(target_dir, filename):
    """ Return a binary String containing the binary data for a tarball of the specified directory """
    print("Preparing the lab directory for transmission...")

    data = BytesIO()
    tar = tarfile.open(filename, "w|bz2", data)

    top_folder_name = os.path.split(target_dir)[1]

    def tar_filter(filename):
        """Returns True if we should tar the file.
        Avoid uploading .pyc files or the .git subdirectory (if any)"""
        if filename in [".git",".DS_Store","__pycache__"]:
            return False
        if os.path.splitext(filename)[1] == ".pyc":
            return False
        return True

    def add_dir(currentDir, t_verbose=False):
        for currentFile in os.listdir(currentDir):
            fullPath=os.path.join(currentDir,currentFile)
            if t_verbose:
                print(currentFile, end=' ')
            if tar_filter(currentFile):
                if t_verbose:
                    print("")
                tar.add(fullPath,arcname=fullPath.replace(target_dir, top_folder_name,1),recursive=False)
                if os.path.isdir(fullPath):
                    add_dir(fullPath)
            elif t_verbose:
                print("....skipped")

    add_dir(target_dir)

    print("Done.")
    print()
    print("The following files will be uploaded:")

    for f in tar.getmembers():
        print(" - {}".format(f.name))

    tar.close()

    return data.getvalue()

def make_test_counter_decorator():
    tests = []
    def make_test(getargs, testanswer, expected_val, name = None, type = 'FUNCTION'):
        if name != None:
            getargs_name = name
        elif not callable(getargs):
            getargs_name = "_".join(getargs[:-8].split('_')[:-1])
            getargs = lambda: getargs
        else:
            getargs_name = "_".join(getargs.__name__[:-8].split('_')[:-1])

        tests.append( ( getargs_name,
                        getargs,
                        testanswer,
                        expected_val,
                        getargs_name,
                        type ) )

    def get_tests():
        return tests

    return make_test, get_tests


make_test, get_tests = make_test_counter_decorator()


if __name__ == '__main__':
    if test_offline():
        print("Local tests passed! Submit your code on Gradescope to have it graded.")

