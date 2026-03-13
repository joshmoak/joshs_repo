import itertools

class AllTen:
    def __init__(self, numbers):
        self.numbers = numbers
        self.operations = ['+', '-', '/', '*']

    def get_all_expressions(self, current_ops, current_nums):
        """Generate all expression strings from nums and op_symbols by all parenthesizations."""
        a, b, c, d = current_nums
        op1, op2, op3 = current_ops
        exprs = []
        # Parenthesization patterns:
        # 1. ((a op1 b) op2 c) op3 d
        exprs.append(f"(({a} {op1} {b}) {op2} {c}) {op3} {d}")
        # 2. (a op1 (b op2 c)) op3 d
        exprs.append(f"({a} {op1} ({b} {op2} {c})) {op3} {d}")
        # 3. (a op1 b) op2 (c op3 d)
        exprs.append(f"({a} {op1} {b}) {op2} ({c} {op3} {d})")
        # 4. a op1 ((b op2 c) op3 d)
        exprs.append(f"{a} {op1} (({b} {op2} {c}) {op3} {d})")
        # 5. a op1 (b op2 (c op3 d))
        exprs.append(f"{a} {op1} ({b} {op2} ({c} {op3} {d}))")
        return exprs
        
    def safe_eval(self, expr):
        """Evaluate expr safely, catch divisions by zero."""
        try:
            val = eval(expr)
            if abs(val - round(val)) < 1e-8:
                val = round(val)
            return val
        except ZeroDivisionError:
            return None
            
    def find_solutions(self, target):
        for perm in set(itertools.permutations(self.numbers)):
            for current_ops in itertools.product(self.operations, repeat=3):
                exprs = self.get_all_expressions(current_ops, perm)
                for expr in exprs:
                    val = self.safe_eval(expr)
                    if val is None:
                        continue
                    if val == target:
                        return expr

    def get_solutions(self):
        solutions = {i: int(eval(self.find_solutions(i)) for i in range(1,11))}
        return solutions
        # solutions = []
        # for i in range(1,11):
        #     sol = self.find_solutions(i)
        #     # print(f'{sol} = {int(eval(sol))}')
        #     solutions.append(f'{sol} = {int(eval(sol))}')
        # self.solutions = solutions
        

def main():
    solver = AllTen(
        numbers = [3,5,5,8]
    )

    solver.get_solutions()
    return solver.solutions

if __name__ == "__main__":
   main()

