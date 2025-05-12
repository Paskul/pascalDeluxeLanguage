import io, sys
import traceback
from sly import Lexer, Parser
from pascalDeluxe import MyLexer, MyParser

tests = [
    ("print head.(tail.[10,20,30])",                      "20"),
    ("print -(head.(tail.[10,20,30]))",                   "-20"),
    ("print 1 + 2",                                       "3"),
    ("print 1 + [1,2]",                                   "SyntaxError"),
    ("print 1 == 1",                                      "True"),
    ("print 1 == [1]",                                    "SyntaxError"),
    ("print True != False",                               "True"),
    ("print True != 1",                                   "SyntaxError"),
    ("print 3 * 4",                                       "12"),
    ("print 3 * [1,2]",                                   "SyntaxError"),
    ("print 5 < 10",                                      "True"),
    ("print 5 < True",                                    "SyntaxError"),
    ("print head.[10,20]",                                "10"),
    ("print head.10",                                     "SyntaxError"),
    ("print tail.[10,20]",                                "[ 20 ]"),
    ("print tail.10",                                     "SyntaxError"),
    ("print -(3+2)",                                      "-5"),
    ("print -[1,2]",                                      "0 - [ 1, 2 ]"),
    ("print sort.[13,51,2,2,5,4]",                        "[ 2, 2, 4, 5, 13, 51 ]"),
    ("print sort.True",                                   "SyntaxError"),
    ("print \"hey \" + \"whats \" + \"up\"",             "hey whats up"),
    ("print \"hey\" + 2",                                 "SyntaxError"),
    ("print True and False",                              "False"),
    ("print True and 2",                                  "SyntaxError"),
    ("print if True and True then 1 else 0 endif",        "1"),
    ("print not (True and False) or False",               "True"),
    ("print 10 % 3",                                      "1"),
    ("print 2 ** 8",                                      "256"),
    ("print 2 * 3 % 4",                                   "2"),
    ("print 10 - 7 % 5",                                  "8"),
    ("print 2 ** 3 ** 2",                                 "512"),
    ("print (2 ** 3) ** 2",                               "64"),
    ("print (3 ** 3) % 5",                                "2"),
    ("print -2 % 3",                                      "1"),
    ("{ x := 5 print x { y := x * x print y } dump }",   "y: Value(int, 25)"),
]

def run_command(cmd):
    lexer  = MyLexer()
    parser = MyParser()
    buf = io.StringIO()
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout = buf
    sys.stderr = buf
    try:
        parser.parse(lexer.tokenize(cmd))
    except Exception:
        tb = traceback.format_exc().strip().splitlines()
        buf.write(tb[-1])
    finally:
        sys.stdout, sys.stderr = old_out, old_err
    return buf.getvalue().strip()

if __name__ == "__main__":
    for cmd, expected in tests:
        output = run_command(cmd)
        print(f">> {cmd}")
        print(output or "<no output>")
        result = "PASS" if expected in output else "FAIL"
        print(f"[{result}] (looking for “{expected}”)")
        print("-" * 60)