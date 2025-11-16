#!/usr/bin/env python3
"""
mini_compiler.py

Simple compiler front-end:
 - Lexical analysis (tokenizer)
 - Recursive-descent parser for a small grammar
 - Syntax error detection + suggestions
 - Two recovery strategies:
     * phrase-level (insert a "virtual" missing token and consume unexpected token to make progress)
     * panic-mode (skip tokens until a synchronizing token like ';' or '}' is found)
 - Pretty printing of AST and errors.

Copy this file and run: python3 mini_compiler.py
"""

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple, Any

# ---------------------------
# Tokens & Lexer
# ---------------------------

@dataclass
class Token:
    type: str
    lexeme: str
    line: int
    col: int
    def __repr__(self):
        return f"{self.type}('{self.lexeme}')@{self.line}:{self.col}"

# Order matters: longer patterns (==, !=) come before single-char '='
TOKEN_SPEC = [
    ("NUMBER",   r"\d+(\.\d+)?"),
    ("EQ",       r"=="),
    ("NEQ",      r"!="),
    ("ID",       r"[A-Za-z_][A-Za-z0-9_]*"),
    ("ASSIGN",   r"="),
    ("SEMI",     r";"),
    ("LPAREN",   r"\("),
    ("RPAREN",   r"\)"),
    ("LBRACE",   r"\{"),
    ("RBRACE",   r"\}"),
    ("PLUS",     r"\+"),
    ("MINUS",    r"-"),
    ("STAR",     r"\*"),
    ("SLASH",    r"/"),
    ("COMMA",    r","),
    ("LT",       r"<"),
    ("GT",       r">"),
    ("WS",       r"[ \t]+"),
    ("NEWLINE",  r"\n"),
    ("UNKNOWN",  r"."),
]

master_pat = re.compile("|".join(f"(?P<{n}>{p})" for n,p in TOKEN_SPEC))
KEYWORDS = {"if","else","while","for","return","int","float","print"}

def lex(source: str) -> List[Token]:
    tokens: List[Token] = []
    line = 1
    line_start = 0
    for mo in master_pat.finditer(source):
        kind = mo.lastgroup
        text = mo.group(kind)
        if kind == "NEWLINE":
            line += 1
            line_start = mo.end()
            continue
        if kind == "WS":
            continue
        col = mo.start() - line_start + 1
        if kind == "ID" and text in KEYWORDS:
            kind = "KEYWORD"
        tokens.append(Token(kind, text, line, col))
    tokens.append(Token("EOF", "", line, 1))
    return tokens

# ---------------------------
# AST Node
# ---------------------------

@dataclass
class ASTNode:
    nodetype: str
    value: Any = None
    children: List[Any] = None
    token: Optional[Token] = None

    def __repr__(self):
        if self.value is not None:
            return f"{self.nodetype}({self.value})"
        return f"{self.nodetype}"

# ---------------------------
# Parser with error handling
# ---------------------------

class Parser:
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.pos = 0
        # errors: tuple(message, token, suggestion)
        self.errors: List[Tuple[str, Token, str]] = []

    def current(self) -> Token:
        return self.tokens[self.pos]

    def advance(self) -> Token:
        tok = self.current()
        if tok.type != "EOF":
            self.pos += 1
        return tok

    def expect(self, expected_type: str, suggestion: Optional[str] = None) -> Token:
        tok = self.current()
        if tok.type == expected_type:
            return self.advance()
        msg = f"Expected {expected_type} but found '{tok.lexeme}' ({tok.type}) at {tok.line}:{tok.col}."
        sugg = suggestion or self._suggest_for_expected(expected_type, tok)
        self._report_error(msg, tok, sugg)
        # Recovery: consume offending token so parser makes progress and return a 'virtual' expected token
        self.advance()
        fake = Token(expected_type, f"<missing {expected_type}>", tok.line, tok.col)
        return fake

    def _report_error(self, msg: str, tok: Token, suggestion: Optional[str]):
        self.errors.append((msg, tok, suggestion or "No suggestion available."))

    def _suggest_for_expected(self, expected_type: str, tok: Token) -> str:
        if expected_type == "SEMI":
            return "Did you forget a ';' at the end of the statement?"
        if expected_type == "RPAREN":
            return "Possible missing ')' — check matching parentheses."
        if expected_type == "RBRACE":
            return "Possible missing '}' — check block delimiters."
        if expected_type == "ID":
            return "Identifier expected. Maybe a variable name is missing."
        return f"Expected {expected_type} here."

    # Synchronizing tokens for panic mode
    SYNC_TOKENS = {"SEMI", "RBRACE", "EOF"}
    def sync(self):
        # Panic-mode: skip tokens until we find a synchronization token
        while self.current().type not in Parser.SYNC_TOKENS:
            self.advance()
        # If we stopped at a semicolon, consume it to continue after the statement
        if self.current().type == "SEMI":
            self.advance()

    # ---------------------------
    # Grammar (commented)
    # program -> stmt_list EOF
    # stmt_list -> (stmt)*
    # stmt -> assignment | expr_stmt | if_stmt | while_stmt | block | print_stmt
    # assignment -> ID ASSIGN expr SEMI
    # expr_stmt -> expr SEMI
    # print_stmt -> 'print' LPAREN expr RPAREN SEMI
    # block -> LBRACE stmt_list RBRACE
    # expr -> term ((PLUS|MINUS) term)*
    # term -> factor ((STAR|SLASH) factor)*
    # factor -> NUMBER | ID | LPAREN expr RPAREN | MINUS factor
    # ---------------------------

    def parse(self) -> ASTNode:
        prog = ASTNode("Program", children=[])
        while self.current().type != "EOF":
            stmt = self.parse_stmt()
            if stmt is None:
                # Something went wrong; try to sync and continue
                self.sync()
                continue
            prog.children.append(stmt)
        return prog

    def parse_stmt(self) -> Optional[ASTNode]:
        tok = self.current()
        if tok.type == "KEYWORD" and tok.lexeme == "print":
            return self.parse_print_stmt()
        if tok.type == "ID":
            # lookahead for assignment
            if self.pos + 1 < len(self.tokens) and self.tokens[self.pos+1].type == "ASSIGN":
                return self.parse_assignment()
            return self.parse_expr_stmt()
        if tok.type == "LBRACE":
            return self.parse_block()
        if tok.type == "KEYWORD" and tok.lexeme in ("if","while"):
            return self.parse_control_stmt()
        if tok.type == "SEMI":
            self.advance()
            return ASTNode("EmptyStmt")
        # Unexpected token at statement level -> report & panic-sync
        self._report_error(f"Unexpected token '{tok.lexeme}' ({tok.type}) at {tok.line}:{tok.col} while parsing statement.", tok, "Remove or fix the token.")
        self.sync()
        return None

    def parse_assignment(self) -> ASTNode:
        id_tok = self.expect("ID")
        self.expect("ASSIGN", suggestion="Assignment operator '=' expected.")
        expr = self.parse_expr()
        self.expect("SEMI", suggestion="Each statement should end with ';'.")
        return ASTNode("Assign", value=id_tok.lexeme, children=[expr], token=id_tok)

    def parse_expr_stmt(self) -> ASTNode:
        expr = self.parse_expr()
        self.expect("SEMI", suggestion="Expression statements must end with ';'.")
        return ASTNode("ExprStmt", children=[expr])

    def parse_print_stmt(self) -> ASTNode:
        self.expect("KEYWORD")  # 'print'
        self.expect("LPAREN", suggestion="print needs '(' before its arguments.")
        expr = self.parse_expr()
        self.expect("RPAREN", suggestion="Missing ')' after print arguments.")
        self.expect("SEMI", suggestion="print statement should end with ';'.")
        return ASTNode("Print", children=[expr])

    def parse_block(self) -> ASTNode:
        self.expect("LBRACE")
        block = ASTNode("Block", children=[])
        while self.current().type not in ("RBRACE", "EOF"):
            s = self.parse_stmt()
            if s:
                block.children.append(s)
        self.expect("RBRACE", suggestion="Missing '}' to close block.")
        return block

    def parse_control_stmt(self) -> ASTNode:
        kw = self.expect("KEYWORD")  # if or while
        self.expect("LPAREN", suggestion=f"{kw.lexeme} needs '(' after it.")
        cond = self.parse_expr()
        self.expect("RPAREN", suggestion="Missing ')' after condition.")
        body = self.parse_stmt()
        return ASTNode(kw.lexeme, children=[cond, body])

    # Expression parsing with precedence
    def parse_expr(self) -> ASTNode:
        node = self.parse_term()
        while self.current().type in ("PLUS", "MINUS"):
            op = self.advance()
            rhs = self.parse_term()
            node = ASTNode("BinaryOp", value=op.lexeme, children=[node, rhs], token=op)
        return node

    def parse_term(self) -> ASTNode:
        node = self.parse_factor()
        while self.current().type in ("STAR", "SLASH"):
            op = self.advance()
            rhs = self.parse_factor()
            node = ASTNode("BinaryOp", value=op.lexeme, children=[node, rhs], token=op)
        return node

    def parse_factor(self) -> ASTNode:
        tok = self.current()
        if tok.type == "NUMBER":
            self.advance()
            return ASTNode("Number", value=tok.lexeme, token=tok)
        if tok.type == "ID":
            self.advance()
            return ASTNode("Variable", value=tok.lexeme, token=tok)
        if tok.type == "LPAREN":
            self.advance()
            expr = self.parse_expr()
            if self.current().type != "RPAREN":
                self._report_error(f"Missing ')' after expression at {tok.line}:{tok.col}.", self.current(), "Add ')' to close parenthesis.")
                # Return the expression anyway (phrase-level recovery: inserted RPAREN virtually)
                return expr
            self.advance()
            return ASTNode("ParenExpr", children=[expr])
        if tok.type == "MINUS":
            self.advance()
            f = self.parse_factor()
            return ASTNode("UnaryOp", value="-", children=[f], token=tok)
        # Unexpected token in factor: report and advance to avoid infinite loop
        self._report_error(f"Unexpected token '{tok.lexeme}' in expression at {tok.line}:{tok.col}.", tok, "Check for missing operands or parentheses.")
        self.advance()
        return ASTNode("Number", value="0", token=tok)

# ---------------------------
# Utilities: pretty print AST + errors
# ---------------------------

def pretty_print_ast(node: ASTNode, indent: int = 0, max_depth: int = 8):
    if node is None:
        return
    if indent > max_depth:
        print("  " * indent + "...")
        return
    spacer = "  " * indent
    if node.value is not None and not node.children:
        print(f"{spacer}{node.nodetype}: {node.value}")
        return
    print(f"{spacer}{node.nodetype}" + (f": {node.value}" if node.value is not None else ""))
    if node.children:
        for c in node.children:
            pretty_print_ast(c, indent + 1, max_depth)

def print_errors(errors: List[Tuple[str, Token, str]]):
    if not errors:
        print("No syntax errors detected.")
        return
    print("\nDetected errors and suggestions:")
    for i, (msg, tok, sugg) in enumerate(errors, 1):
        print(f"{i}. {msg}\n   At token: {tok}\n   Suggestion: {sugg}\n")

# ---------------------------
# Demo CLI
# ---------------------------

def run_demo(src: str):
    print("SOURCE:\n" + src)
    tokens = lex(src)
    print("\nTOKENS (first 80):")
    print(tokens[:80])
    parser = Parser(tokens)
    ast = parser.parse()
    print("\nAST:")
    pretty_print_ast(ast)
    print_errors(parser.errors)

if __name__ == "__main__":
    samples = [
        # 1) Correct program
        ("Correct example",
         "x = 3 + 4; y = x * (2 + 5); print(y);"),
        # 2) Missing semicolon and missing ')'
        ("Missing semicolon & paren",
         "x = 3 + 4\n y = x * (2 + 5; print(y)"),
        # 3) Operator without operand
        ("Operator without operand",
         "a = + 5; b = 7 * ;"),
    ]
    for title, src in samples:
        print("\n" + "="*60)
        print("Demo:", title)
        print("="*60)
        run_demo(src)
