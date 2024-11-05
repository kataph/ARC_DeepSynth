import random
import numpy as np
import os
import sys

import vose

from ARC_type_system import Type, PolymorphicType, PrimitiveType, Arrow, UnknownType, INT, BOOL
from ARC_program import Program, Function, Variable, BasicPrimitive, New

# make sure hash is deterministic
hashseed = os.getenv('PYTHONHASHSEED')
if not hashseed:
    os.environ['PYTHONHASHSEED'] = '0'
    os.execv(sys.executable, [sys.executable] + sys.argv)

class ARC_PCFG:
    """
    Object that represents a probabilistic context-free grammar
    with normalised weights

    rules: a dictionary of type {S: D}
    with S a non-terminal and D a dictionary : {P : l, w}
    with P a program, l a list of non-terminals, and w a weight
    representing the derivation S -> P(S1, S2, ...) with weight w for l' = [S1, S2, ...]

    list_derivations: a dictionary of type {S: l}
    with S a non-terminal and l the list of programs P appearing in derivations from S,
    sorted from most probable to least probable

    max_probability: a dictionary of type {S: (Pmax, probability)} cup {(S, P): (Pmax, probability)}
    with S a non-terminal

    hash_table_programs: a dictionary {hash: P}
    mapping hashes to programs
    for all programs appearing in max_probability
    """

    def __init__(self, start, rules, max_program_depth, clean = False):
        self.start = start
        self.rules = rules
        self.max_program_depth = max_program_depth

        self.hash = hash(str(rules))

        if clean:
            self.remove_non_productive()
            self.remove_non_reachable()
            self.normalise()
            self.sort()

    def type_request(self) -> Type:
        type_req = self.start[0]
        variables = []
        for S in self.rules:
            for P in self.rules[S]:
                if isinstance(P, Variable):
                    if P not in variables:
                        variables.append(P)
        n = len(variables)
        for i in range(n):
            j = n - i - 1
            for v in variables:
                if v.variable == j:
                    type_req = Arrow(v.type, type_req)
        return type_req

    def __hash__(self):
        return self.hash

    def __str__(self):
        s = "Print a PCFG\n"
        s += "start: {}\n".format(self.start)
        for S in reversed(self.rules):
            s += "#\n {}\n".format(S)
            for P in self.rules[S]:
                args_P, w = self.rules[S][P]
                s += "   {} - {}: {}     {}\n".format(P, P.type, args_P, w)
        return s

    def init_vose(self):
        self.vose_samplers = {}
        self.list_derivations = {}

        for S in self.rules:
            self.list_derivations[S] = sorted(
                self.rules[S], key=lambda P: self.rules[S][P][1]
            )
            self.vose_samplers[S] = vose.Sampler(
                np.array([self.rules[S][P][1] for P in self.list_derivations[S]],dtype=float)
            )

    def sort(self):
        for S in self.rules:
            sorted_derivation_list = sorted(
                self.rules[S], key=lambda P: -self.rules[S][P][1]
            )
            new_rules = {}
            for P in sorted_derivation_list:
                new_rules[P] = self.rules[S][P]
            self.rules[S] = new_rules

    def normalise(self):
        for S in self.rules:
            s = sum([self.rules[S][P][1] for P in self.rules[S]])
            for P in list(self.rules[S].keys()):
                args_P, w = self.rules[S][P]
                self.rules[S][P] = (args_P, w / s)

    def return_unique(self, P):
        """
        ensures that if a program appears in several rules,
        it is represented by the same object
        """
        if P.hash in self.hash_table_programs:
            return self.hash_table_programs[P.hash]
        else:
            self.hash_table_programs[P.hash] = P
            return P

    def remove_non_productive(self):
        """
        remove non-terminals which do not produce programs
        """
        new_rules = {}
        for S in reversed(self.rules):
            for P in self.rules[S]:
                args_P, w = self.rules[S][P]
                if all([arg in new_rules for arg in args_P]) and w > 0:
                    if S not in new_rules:
                        new_rules[S] = {}
                    new_rules[S][P] = self.rules[S][P]

        for S in set(self.rules):
            if S in new_rules:
                self.rules[S] = new_rules[S]
            else:
                del self.rules[S]

    def remove_non_reachable(self):
        """
        remove non-terminals which are not reachable from the initial non-terminal
        """
        reachable = set()
        reachable.add(self.start)

        reach = set()
        new_reach = set()
        reach.add(self.start)

        for i in range(self.max_program_depth):
            new_reach.clear()
            for S in reach:
                for P in self.rules[S]:
                    args_P, _ = self.rules[S][P]
                    for arg in args_P:
                        new_reach.add(arg)
                        reachable.add(arg)
            reach.clear()
            reach = new_reach.copy()

        for S in set(self.rules):
            if S not in reachable:
                del self.rules[S]

    def compute_max_probability(self):
        """
        populates a dictionary max_probability
        """
        self.hash_table_programs = {}
        self.max_probability = {}

        for S in reversed(self.rules):
            best_program = None
            best_probability = 0

            for P in self.rules[S]:
                args_P, w = self.rules[S][P]
                P_unique = self.return_unique(P)

                if len(args_P) == 0:
                    self.max_probability[(S, P)] = P_unique
                    P_unique.probability[(self.__hash__(), S)] = w
                    # assert P_unique.probability[
                    #     (self.__hash__(), S)
                    # ] == self.probability_program(S, P_unique)

                else:
                    new_program = Function(
                        function=P_unique,
                        arguments=[self.max_probability[arg] for arg in args_P],
                        type_=S[0],
                        probability={},
                    )
                    P_unique = self.return_unique(new_program)
                    probability = w
                    for arg in args_P:
                        probability *= self.max_probability[arg].probability[(self.__hash__(), arg)]
                    self.max_probability[(S, P)] = P_unique
                    # assert (self.__hash__(), S) not in P_unique.probability
                    P_unique.probability[(self.__hash__(), S)] = probability
                    # assert probability == self.probability_program(S, P_unique)

                if (
                    self.max_probability[(S, P)].probability[(self.__hash__(), S)]
                    > best_probability
                ):
                    best_program = self.max_probability[(S, P)]
                    best_probability = self.max_probability[(S, P)].probability[
                        (self.__hash__(), S)
                    ]

            # assert best_probability > 0
            self.max_probability[S] = best_program

    def sampling(self):
        """
        A generator that samples programs according to the PCFG G
        """
        self.init_vose()

        while True:
            yield self.sample_program(self.start)

    def sample_program(self, S):
        i = self.vose_samplers[S].sample()
        P = self.list_derivations[S][i]
        args_P, w = self.rules[S][P]
        if len(args_P) == 0:
            return P
        arguments = []
        for arg in args_P:
            arguments.append(self.sample_program(arg))
        return Function(P, arguments)

    def probability_program(self, S, P):
        """
        Compute the probability of a program P generated from the non-terminal S
        """
        if isinstance(P, Function):
            F = P.function
            args_P = P.arguments
            probability = self.rules[S][F][1]
            
            for i, arg in enumerate(args_P):
                probability *= self.probability_program(self.rules[S][F][0][i], arg)
            return probability

        if isinstance(P, (Variable, BasicPrimitive, New)):
            return self.rules[S][P][1]

        print("probability_program", P)
        assert False

    def get_sbsur_sampler(self, S=None, seed=None):
        """
        Return an sbs ur sampler from this PCFG starting from non-terminal S or from start if S is None.
        SBSUR won't return anything if the PCFG allows only one program.
        Returns a function: batch_size -> list[program]
        """
        from sbsur import SequenceGenerator, sample

        # Build the list of derivations
        try:
            self.list_derivations
        except:
            self.list_derivations = {}
            for K in self.rules:
                self.list_derivations[K] = sorted(
                    self.rules[K], key=lambda P: self.rules[K][P][1]
                )

        max_categories = max(len(self.list_derivations[J]) for J in self.rules)
        S = S or self.start

        # int list -> log probs | None
        def get_logprobs(sequence):
            context_stack = [S]
            for i in sequence:
                current = context_stack.pop()
                # Skip when there's only 1 possibility since no sampling is necessary
                # Since the grammar is correctly defined we should never pop an empty stack
                while len(self.list_derivations[current]) == 1:
                    P = self.list_derivations[current][0]
                    args_P, _ = self.rules[current][P]
                    for arg in args_P:
                        context_stack.append(arg)
                    current = context_stack.pop()

                # Get the derivation
                P = self.list_derivations[current][i]
                args_P, _ = self.rules[current][P]
                for arg in args_P:
                    context_stack.append(arg)
            # If this is a valid program => No further sampling is required
            if len(context_stack) == 0:
                return None
            # Pop the current context
            current = context_stack.pop()
            # If there's only 1 derivation skip
            while len(self.list_derivations[current]) == 1:
                P = self.list_derivations[current][0]
                args_P, _ = self.rules[current][P]
                for arg in args_P:
                        context_stack.append(arg)
                if not context_stack:
                    # Reached terminal node
                    return None
                current = context_stack.pop()
            # Give log probs
            return np.log(np.array([self.rules[current][P][1] for P in self.list_derivations[current]], dtype=float))

        gen = SequenceGenerator(lambda x:[get_logprobs(el) for el in x], max_categories, seed)

        # int list -> Program cons list
        def seq2prog(sequence):
            context_stack = [S]
            # Stack of functions
            call_stack = []
            # Stack of valid programs
            program = None
            for i in sequence:
                current = context_stack.pop()
                # We need to manage cases when there's only 1 derivation possible because we don't need sampling
                while len(self.list_derivations[current]) == 1:
                    P = self.list_derivations[current][0]
                    args_P, w = self.rules[current][P]
                    program = (P, program)
                    for arg in args_P:
                        context_stack.append(arg)
                    current = context_stack.pop()

                P = self.list_derivations[current][i]
                args_P, w = self.rules[current][P]
                program = (P, program)
                for arg in args_P:
                    context_stack.append(arg)
            # Context stack may contain potentially a lot of calls with 1 possible derivation
            while context_stack:
                current = context_stack.pop()
                assert len(self.list_derivations[current]) == 1, f"Current: {current} has more than 1 derivation:{self.list_derivations[current]}"
                P = self.list_derivations[current][0]
                args_P, w = self.rules[current][P]
                program = (P, program)
                for arg in args_P:
                    context_stack.append(arg)
            assert not call_stack
            return program

        def sampler(batch_size):
            if gen.is_exhausted():
                return []
            sequences = sample(gen, batch_size)
            return [seq2prog(seq) for seq in sequences]

        return sampler
