from sly import Lexer, Parser
from colorama import Fore, Style
import copy

DEBUG = True

def printCyan(s):
	print(Fore.CYAN, end='')
	print(s)
	print(Style.RESET_ALL, end='')

def printBlue(s):
	print(Fore.BLUE, end='')
	print(s)
	print(Style.RESET_ALL, end='')

def printGreen(s):
	print(Fore.GREEN, end='')
	print(s)
	print(Style.RESET_ALL, end='')

def printMagenta(s):
	print(Fore.MAGENTA, end='')
	print(s)
	print(Style.RESET_ALL, end='')

def printYellow(s):
	print(Fore.YELLOW, end='')
	print(s)
	print(Style.RESET_ALL, end='')

def apply(operation, lhs, rhs):
	if operation == '+':
		return lhs + rhs
	elif operation == '-':
		return lhs - rhs
	elif operation == '*':
		return lhs * rhs
	elif operation == '/':
		return lhs // rhs
	elif operation == '%':
		return lhs % rhs
	elif operation == '**':
		return lhs ** rhs
	elif operation == '==':
		return lhs == rhs
	elif operation == '!=':
		return lhs != rhs
	elif operation == '>':
		return lhs > rhs
	elif operation == '>=':
		return lhs >= rhs
	elif operation == '<':
		return lhs < rhs
	elif operation == '<=':
		return lhs <= rhs
	elif operation in ('and', '&&'):
		return lhs and rhs
	elif operation in ('or', '||'):
		return lhs or rhs

class Value:
	# Stores values of variables and expressions
	def __init__(self, dataType, components):
		# Components is a dictionary of the values, expressions or other
		# data needed to define a value
		self.dataType = dataType
		
		if DEBUG: printMagenta(f'Value construction {dataType} {repr(components)}')
		
		if dataType in ['int', 'bool', 'string']:
			self.value = components['value']
		elif dataType == 'expr':
			self.expression = components['expression']
		elif dataType == 'expr_list':
			self.expressions = components['expressions']
		elif dataType == 'function':
			self.variable = components['variable']
			self.result = components['result']
		elif dataType == 'operation':
			self.lhs = components['lhs']
			self.operation = components['operation']
			self.rhs = components['rhs']
		elif dataType == 'id':
			self.id = components['id']
		elif dataType == 'conditional':
			self.condition = components['condition']
			self.then_clause = components['then_clause']
			self.else_clause = components['else_clause']
		elif dataType == 'list':
			self.elements = components['elements']
		else:
			print(f'ERROR: constructor fall through {dataType} {repr(components)}')
			
		if DEBUG: printMagenta(f'Constructor result {repr(self)}')
			
	def replace(self, variable, value, valueLookup):
		# Make a copy and perform appropriate substitutions
		if self.dataType in ['int','bool']:
			v = Value(self.dataType, {'value': self.value})
		elif self.dataType == 'expr':
			v = Value('expr', {'expr': self.expr.replace(variable, value, valueLookup)})
		elif self.dataType == 'expr_list':
			newExpressions = [e.replace(variable, value, valueLookup) for e in self.expressions]
			v = Value('expr_list', {'expressions': newExpressions})
		elif self.dataType == 'function':
			v = Value('function', {'variable': str(self.variable), 'result': self.result.replace(variable, value, valueLookup)})
		elif self.dataType == 'operation':
			v = Value('operation', {'lhs': self.lhs.replace(variable, value, valueLookup), 'operation': self.operation, 'rhs': self.rhs.replace(variable, value, valueLookup)})
		elif self.dataType == 'id':
			if self.id == variable:
				v = copy.deepcopy(value)
			else:
				v = Value('id', {'id': self.id})
		elif self.dataType == 'conditional':
			condition = self.condition.replace(variable, value, valueLookup).simplify(valueLookup)
			if condition.dataType == 'bool':
				if condition.value:
					v = self.then_clause.replace(variable, value, valueLookup).simplify(valueLookup)
				else:
					v = self.else_clause.replace(variable, value, valueLookup).simplify(valueLookup)
			else:
				then_clause = self.then_clause.replace(variable, value, valueLookup)
				else_clause = self.else_clause.replace(variable, value, valueLookup)
				v = Value('conditional', {'condition': condition, 'then_clause': then_clause, 'else_clause': else_clause})
		elif self.dataType == 'list':			
			newElements = [e.replace(variable, value, valueLookup) for e in self.elements]
			v = Value('list', {'elements': newElements})
		else:
			print('ERROR:  fall through in replace')
		
		if DEBUG and str(self) != str(v): printCyan(f'Replace {variable} in {repr(self)} with {repr(value)} to get {repr(v)}') 
		return v
		
	def simplify(self, valueLookup):
		if DEBUG:
			printYellow(f'Simplifying {repr(self)}')
			
		if self.dataType in ['int','bool']:
			v = Value(self.dataType, {'value': self.value})
		elif self.dataType == 'expr':
			v = Value('expr', {'expr': self.expression.simplify(valueLookup)})
		elif self.dataType == 'expr_list':
			newExpressions = [] # [e.simplify() for e in self.expressions]
			for i, e in enumerate(self.expressions):
				newExpressions.append(e.simplify(valueLookup))
			i = 0
			while i < len(newExpressions)-1:				
				if newExpressions[i].dataType == 'function' and newExpressions[i+1].dataType in ['int','bool']:
					prefix = newExpressions[:i]
					suffix = newExpressions[i+2:]
					clause = newExpressions[i].result.replace(newExpressions[i].variable, newExpressions[i+1], valueLookup)
					newExpressions = prefix + [clause] + suffix
				elif newExpressions[i].dataType == 'id' and newExpressions[i].id in valueLookup and valueLookup[newExpressions[i].id].dataType == 'function' and newExpressions[i+1].dataType in ['int','bool']:
					function = valueLookup[newExpressions[i].id]
					variable = function.variable
					result = function.result
					prefix = newExpressions[:i]
					suffix = newExpressions[i+2:]
					clause = result.replace(variable, newExpressions[i+1], valueLookup)
					newExpressions = prefix + [clause] + suffix
				else:
					i += 1
			if len(newExpressions) == 1:
				v = newExpressions[0].simplify(valueLookup)
			else:
				v = Value('expr_list', {'expressions': newExpressions})
		elif self.dataType == 'function':
			v = Value('function', {'variable': str(self.variable), 'result': self.result.simplify(valueLookup)})
		elif self.dataType == 'operation':
			if self.lhs.dataType == self.rhs.dataType and self.lhs.dataType in ['int', 'bool']:
				result = apply(self.operation, self.lhs.value, self.rhs.value)
				if self.operation in ['+', '-', '/', '*']:
					v = Value(self.lhs.dataType, {'value': result})
				else:
					v = Value('bool', {'value': result})
			else:
				v = Value('operation', {'lhs': self.lhs.simplify(valueLookup), 'operation': self.operation, 'rhs': self.rhs.simplify(valueLookup)})
		elif self.dataType == 'id':
			v = Value('id', {'id': self.id})
		elif self.dataType == 'conditional':
			condition = self.condition.simplify(valueLookup)
			if condition.dataType == 'bool':
				if condition.value:
					v = self.then_clause.simplify(valueLookup)
				else:
					v = self.else_clause.simplify(valueLookup)
			else:
				v = Value('conditional', {'condition': condition, 'then_clause': self.then_clause, 'else_clause': self.else_clause})
		elif self.dataType == 'list':			
			newElements = [e.simplify(valueLookup) for e in self.elements]
			v = Value('list', {'elements': newElements})
		else:
			print('ERROR:  fall through in simplify')
			
		if DEBUG: printYellow(f'Simplified {repr(self)} to {repr(v)}') 
		return v
		
	def __str__(self):
		if self.dataType in ['int', 'bool', 'string']:
			s = str(self.value)
		elif self.dataType == 'expr':
			s = str(self.expr)
		elif self.dataType == 'expr_list':
			s = '(' + ' . '.join(str(e) for e in self.expressions) + ')'
		elif self.dataType == 'operation':
			s = ' '.join(['(', str(self.lhs), self.operation, str(self.rhs), ')'])
		elif self.dataType == 'id':
			s = self.id		
		elif self.dataType == 'conditional':
			s = f'if {self.condition} then {self.then_clause} else {self.else_clause}'
		elif self.dataType == 'function':
			s = f'\\ {self.variable} => {(self.result)}'
		elif self.dataType == 'list':
			s = f'[ {", ".join(str(e) for e in self.elements)} ]'
			
		while s.startswith('((') and s.endswith('))'):
			s = s[1:-1]
			
		return s
		
	def __repr__(self):
		if self.dataType == 'int':
			return f'Value(int, {self.value})'
		elif self.dataType == 'bool':
			return f'Value(bool, {self.value})'
		elif self.dataType == 'string':
			return f'Value(string, {self.value})'
		elif self.dataType == 'expr':
			return f'Value(expr, {repr(self.expr)})'
		elif self.dataType == 'expr_list':
			return f'Value(expr_list, [{", ".join(repr(e) for e in self.expressions)}])'
		elif self.dataType == 'operation':
			return f'Value(operation, {repr(self.lhs)}, {self.operation}, {repr(self.rhs)})'
		elif self.dataType == 'id':
			return f'Value(id, {self.id})'
		elif self.dataType == 'conditional':
			return f'Value(conditional, {repr(self.condition)}, {repr(self.then_clause)}, {repr(self.else_clause)})'
		elif self.dataType == 'function':
			return f'Value(function, {self.variable}, {repr(self.result)})'
		elif self.dataType == 'list':
			return f'[ {", ".join(repr(e) for e in self.elements)} ]'

class MyLexer(Lexer):
	# Set of token names.   This is always required
	tokens = { 'NUMBER', 'ID', 'STRING',
			   'ADD_OP', 'MULT_OP', 'ASSIGN',
			   'LPAREN', 'RPAREN', 'SEP', 'ARROW', 'LAMBDA',
			   'EQUAL_OP', 'COMPARE_OP',
			   'PRINT', 'DUMP',
			   'IF', 'THEN', 'ELSE', 'ENDIF',
			   'LBRACKET', 'RBRACKET', 'COMMA',
			   'HEAD', 'TAIL', 'SORT',
			   'AND_OP', 'OR_OP', 'NOT_OP',
			   'MOD_OP', 'EXP_OP' }

	# String containing ignored characters
	ignore = ' \t'

	# Regular expression rules for tokens
	ASSIGN  = r':='
	LPAREN	= r'\('
	RPAREN	= r'\)'
	SEP		= r'\.'
	ARROW	= r'=>'
	LAMBDA	= r'\\'
	ADD_OP	= r'\+|-'
	EXP_OP  = r'\*\*'
	MULT_OP = r'\*|/'
	MOD_OP  = r'%'	
	EQUAL_OP	= r'==|!='
	COMPARE_OP	= r'>=|<=|>|<'
	LBRACKET	= r'\['
	RBRACKET	= r'\]'
	COMMA	= r','

	@_(r'&&|and')
	def AND_OP(self, t):
		return t

	@_(r'\|\||or')
	def OR_OP(self, t):
		return t

	@_(r'!|not')
	def NOT_OP(self, t):
		return t
	
	@_(r'"[^"\n]*"')
	def STRING(self, t):
		# get rid of quotes for now
		t.value = t.value[1:-1]
		return t

	@_(r'\d+')
	def NUMBER(self, t):
		t.value = int(t.value)
		return t

	# Identifiers and keywords
	ID = r'[a-zA-Z_][a-zA-Z0-9_]*'
	
	ID['print'] = 'PRINT'
	ID['dump'] = 'DUMP'
	ID['if'] = 'IF'
	ID['else'] = 'ELSE'
	ID['then'] = 'THEN'
	ID['endif'] = 'ENDIF'
	ID['head'] = 'HEAD'
	ID['tail'] = 'TAIL'
	ID['sort']  = 'SORT'
	ID['and'] = 'AND_OP'
	ID['or'] = 'OR_OP'
	ID['not'] = 'NOT_OP'

	ignore_comment = r'\#.*'

	# Line number tracking
	@_(r'\n+')
	def ignore_newline(self, t):
		self.lineno += t.value.count('\n')

	def error(self, t):
		print('Line %d: Bad character %r' % (self.lineno, t.value[0]))
		self.index += 1

class MyParser(Parser):
	def __init__(self):
		Parser.__init__(self)
		self._values = {}
		
	debugfile = 'parser.out'
	
	# Get the token list from the lexer (required)
	tokens = MyLexer.tokens

	# tighten binding: not > and > or > comparisons > equality > add > multiply
	precedence = (
		('right', 'NOT_OP'),
		('left',  'AND_OP'),
		('left',  'OR_OP'),
		('nonassoc', 'COMPARE_OP'),
		('nonassoc', 'EQUAL_OP'),
		('right','EXP_OP'),
		('left','MULT_OP','MOD_OP'),
		('left',  'ADD_OP'),
	)

	# Grammar rules and actions
	@_('ID ASSIGN expr_list')
	def statement(self, p):
		printGreen(f'Rule: statement => ID ASSIGN expr_list ({p.ID}, {repr(p.expr_list)})')
		self._values[p.ID] = p.expr_list
		
	@_('PRINT expr_list')
	def statement(self, p):
		s = str(p.expr_list)
		while s.startswith('(') and s.endswith(')'):
			s = s[1:-1]
		print(s)
		
	@_('DUMP')
	def statement(self, p):
		for k, v in self._values.items():
			print(f'{k}: {repr(v)}')
			
	@_('expr SEP expr_list')
	def expr_list(self, p):
		if p.expr_list.dataType == 'expr_list':
			v = Value('expr_list', {'expressions': [p.expr] + p.expr_list.expressions})			
		else:
			v = Value('expr_list', {'expressions': [p.expr, p.expr_list]})
		v = v.simplify(self._values)
		
		if DEBUG: printGreen(f'Rule: expr_list -> expr SEP expr_list ({repr(v)})')
		return v
		
	@_('expr')
	def expr_list(self, p):
		if DEBUG: printGreen(f'Rule: expr_list -> expr ({p.expr})')
		return p.expr
		
	@_('LAMBDA ID ARROW expr_list')
	def expr(self, p):
		v = Value('function', {'variable': p.ID, 'result': p.expr_list})
		
		if DEBUG: printGreen(f'Rule: expr -> LAMBDA ID ARROW expr_list ({repr(v)})')
		return v
	
	@_('expr EXP_OP expr')
	def expr(self, p):
		if p.expr0.dataType!='int' or p.expr1.dataType!='int':
			raise SyntaxError(f"TypeError: '**' requires two ints, got {p.expr0.dataType} and {p.expr1.dataType}")
		v = Value('int',{'value': apply(p.EXP_OP,p.expr0.value,p.expr1.value)})
		return v
	

	@_('expr AND_OP expr')
	def expr(self, p):
		# static type check:both bools
		if p.expr0.dataType != 'bool' or p.expr1.dataType != 'bool':
			raise SyntaxError(f"TypeError: 'and' requires two bools, got {p.expr0.dataType} and {p.expr1.dataType}")
		result = apply(p.AND_OP, p.expr0.value, p.expr1.value)
		v = Value('bool', {'value': result})
		if DEBUG: printGreen(f"Rule: expr -> expr AND_OP expr ({repr(v)})")
		return v
	
	@_('expr OR_OP expr')
	def expr(self, p):
		if p.expr0.dataType != 'bool' or p.expr1.dataType != 'bool':
			raise SyntaxError(f"TypeError: 'or' requires two bools, got {p.expr0.dataType} and {p.expr1.dataType}")
		result = apply(p.OR_OP, p.expr0.value, p.expr1.value)
		v = Value('bool', {'value': result})
		if DEBUG: printGreen(f"Rule: expr -> expr OR_OP expr ({repr(v)})")
		return v
	
	@_('NOT_OP expr')
	def expr(self, p):
		if p.expr.dataType != 'bool':
			raise SyntaxError(f"TypeError: 'not' requires a bool, got {p.expr.dataType}")
		v = Value('bool', {'value': not p.expr.value})
		if DEBUG: printGreen(f"Rule: expr -> NOT_OP expr ({repr(v)})")
		return v

	@_('expr ADD_OP term')
	def expr(self, p):
		op = p[1]
		if op == '+' and p.expr.dataType == 'string' and p.term.dataType == 'string':
			v = Value('string', {'value': p.expr.value + p.term.value})
		else:
			if p.expr.dataType != 'int' or p.term.dataType != 'int':
				raise SyntaxError(
					f"TypeError: '{op}' requires two ints or two strings, got "
					f"{p.expr.dataType} and {p.term.dataType}"
				)
			result = apply(op, p.expr.value, p.term.value)
			v = Value('int', {'value': result})
		if DEBUG: printGreen(f"Rule: expr -> expr ADD_OP term ({repr(v)})")
		return v

	@_('term')
	def expr(self, p):
		if DEBUG: printGreen(f'Rule: expr -> term ({p.term})')
		return p.term
	
	@_('IF expr THEN expr ELSE expr ENDIF')
	def expr(self, p):
		if p.expr0.dataType == 'bool':
			if p.expr0.value:
				v = p.expr1
			else:
				v = p.expr2
		else:
			v = Value('conditional', {'condition': p.expr0, 'then_clause': p.expr1, 'else_clause': p.expr2})

		if DEBUG: printGreen(f'Rule: expr -> IF expr THEN expr ELSE expr ENDIF ({repr(v)})')
		return v
			
	@_('term EQUAL_OP term')
	def expr(self, p):
        # static type check: == and != only on same-type int/bool
		if p.term0.dataType not in ['int','bool'] or p.term1.dataType != p.term0.dataType:
			raise SyntaxError(
				f"TypeError: '{p[1]}' requires two {p.term0.dataType}s, got {p.term0.dataType} and {p.term1.dataType}"
			)
		result = apply(p[1], p.term0.value, p.term1.value)
		v = Value('bool', {'value': result})
		if DEBUG: printGreen(f'Rule: expr -> term EQUAL_OP term ({repr(v)})')
		return v

		
	@_('term COMPARE_OP term')
	def expr(self, p):
		# static type check: comparison ops only on ints
		if p.term0.dataType != 'int' or p.term1.dataType != 'int':
			raise SyntaxError(
				f"TypeError: '{p[1]}' requires two ints, got {p.term0.dataType} and {p.term1.dataType}"
			)
		result = apply(p[1], p.term0.value, p.term1.value)
		v = Value('bool', {'value': result})
		if DEBUG: printGreen(f'Rule: expr -> term COMPARE_OP term ({repr(v)})')
		return v
	
	@_('term MOD_OP factor')
	def term(self, p):
		# static type‐check: % only on ints
		if p.term.dataType != 'int' or p.factor.dataType != 'int':
			raise SyntaxError(
				f"TypeError: '%' requires two ints, got {p.term.dataType} and {p.factor.dataType}"
			)
		result = apply(p.MOD_OP, p.term.value, p.factor.value)
		v = Value('int', {'value': result})
		if DEBUG: printGreen(f"Rule: term -> term MOD_OP factor ({repr(v)})")
		return v

	@_('term MULT_OP factor')
	def term(self, p):
		# static type check: * and / only on ints
		if p.term.dataType != 'int' or p.factor.dataType != 'int':
			raise SyntaxError(
				f"TypeError: '{p[1]}' requires two ints, got {p.term.dataType} and {p.factor.dataType}"
			)
		result = apply(p[1], p.term.value, p.factor.value)
		v = Value('int', {'value': result})
		if DEBUG: printGreen(f'Rule: term -> term MULT_OP factor ({repr(v)})')
		return v

	@_('factor')
	def term(self, p):
		if DEBUG: printGreen(f'Rule: term -> factor ({repr(p.factor)})')
		return p.factor

	@_('NUMBER')
	def factor(self, p):
		if DEBUG: printGreen(f'Rule: factor -> NUMBER ({p.NUMBER})')
		return Value('int', {'value': p.NUMBER})
	
	@_('STRING')
	def factor(self, p):
		if DEBUG: printGreen(f"Rule: factor -> STRING ({p.STRING!r})")
		return Value('string', {'value': p.STRING})

	@_('ID')
	def factor(self, p):
		if p.ID == 'True':
			if DEBUG: printGreen('Rule: TRUE -> factor (True)')
			return Value('bool', {'value': True})
		if p.ID == 'False':
			if DEBUG: printGreen('Rule: FALSE -> factor (False)')
			return Value('bool', {'value': False})
		
		if p.ID in self._values:
			if DEBUG: printGreen(f'Rule: id -> factor ({p.ID}, {self._values[p.ID]})')
			return self._values[p.ID]
		else:
			if DEBUG: printGreen(f'Rule: id -> factor ({p.ID})')
			return Value('id', {'id': p.ID})

	@_('LPAREN expr_list RPAREN')
	def factor(self, p):
		if DEBUG: printGreen(f'Rule: LPAREN expr_list RPAREN ({p.expr_list})')
		return p.expr_list
	
	@_('list')
	def factor(self, p):
		if DEBUG: printGreen(f'Rule: list -> factor ({repr(p.list)})')
		return p.list
		
	@_('list')
	def expr(self, p):
		if DEBUG: printGreen(f'Rule: list -> expr ({repr(p.list)})')
		return p.list
		
	@_('LBRACKET RBRACKET')
	def list(self, p):
		if DEBUG: printGreen('Rule: LBRACKET RBRACKET -> list ([])')
		return Value('list', {'elements': []})
		
	@_('LBRACKET comma_sep_list RBRACKET')
	def list(self, p):
		if DEBUG: printGreen(f'Rule: LBRACKET comma_sep_list RBRACKET -> list ({repr(p.comma_sep_list)})')
		return p.comma_sep_list
		
	@_('expr')
	def comma_sep_list(self, p):
		if DEBUG: printGreen(f'Rule: expr -> comma_sep_list ([{repr(p.expr)}])')
		return Value('list', {'elements': [p.expr]})
		
	@_('expr COMMA comma_sep_list')
	def comma_sep_list(self, p):
		v = Value('list', {'elements': [p.expr] + p.comma_sep_list.elements})
		if DEBUG: printGreen(f'Rule: expr COMMA comma_sep_list -> comma_sep_list ({repr(v)})')
		return v
	
	@_('HEAD SEP factor')
	def factor(self, p):
		lst = p.factor
		# static type check: head only on lists
		if lst.dataType != 'list':
			raise SyntaxError(f"TypeError: head expects a list, got {lst.dataType}")
		v = lst.elements[0]
		if DEBUG: printGreen(f'Rule: HEAD SEP factor -> factor ({repr(v)})')
		return v
	
	@_('TAIL SEP factor')
	def factor(self, p):
		lst = p.factor
		# static type check: tail only on lists
		if lst.dataType != 'list':
			raise SyntaxError(f"TypeError: tail expects a list, got {lst.dataType}")
		v = Value('list', {'elements': lst.elements[1:]})
		if DEBUG: printGreen(f'Rule: TAIL SEP factor -> factor ({repr(v)})')
		return v
	
	@_('SORT SEP factor')
	def factor(self, p):
		lst = p.factor
		# static type check: sort only on lists
		if lst.dataType != 'list':
			raise SyntaxError(f"TypeError: sort expects a list, got {lst.dataType}")
		for e in lst.elements:
			if e.dataType != 'int':
				raise SyntaxError("TypeError: sort only supports lists of ints")
		sorted_vals = sorted(e.value for e in lst.elements)
		new_list = [Value('int', {'value': x}) for x in sorted_vals]
		v = Value('list', {'elements': new_list})
		if DEBUG: printGreen(f'Rule: SORT SEP factor -> factor ({repr(v)})')
		return v

	@_('ADD_OP factor')
	def factor(self, p):
		op = p.ADD_OP
		operand = p.factor
		# unary minus
		if op == '-':
			if operand.dataType == 'int':
				v = Value('int', {'value': -operand.value})
			else:
				zero = Value('int', {'value': 0})
				v = Value('operation', {
					'lhs': zero,
					'operation': '-',
					'rhs': operand
				})
		# unary plus
		else:
			v = operand
		if DEBUG:
			printGreen(f'Rule: factor -> ADD_OP factor ({repr(v)})')
		return v

if __name__ == '__main__':
	lexer = MyLexer()
	parser = MyParser()

	while True:
		try:
			text = input('>> ')
			for t in lexer.tokenize(text):
				printBlue(t)
			result = parser.parse(lexer.tokenize(text))
		except EOFError:
			break