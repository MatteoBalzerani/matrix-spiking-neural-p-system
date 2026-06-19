"""Spike related utils classes"""
from texttable import Texttable

class SpikeEvent:
    """contains originator neuron, charge and target neurons"""
    def __init__(self, nid, charge, targets):
        self.nid = nid
        self.charge = charge
        self.targets = targets

class TransformationRule:
    """
    # regex are in the form: a^mod (a^div)^*
    """
    def __init__(self, div, mod, source, target, delay):
        self.div = div # looped "a" in the regex
        self.mod = mod # basic "a" in the regex
        self.source = source # spike consumed when the rule applies
        self.target = target # spike fired when the rule applies
        self.delay = delay # refractory period of the rules

    def check(self, charge):
        #with div and mod is possible to manage all value condition for charge
        if charge < 0 and (self.div < 0 or self.mod < 0): #neuron contains anti-spikes
            if self.div == 0:
                return charge == self.mod
            else:
                return (self.mod - charge) % -self.div == 0
        if charge > 0 and charge >= self.mod and charge >= self.target: #for avoid negative values
            if self.div > 0:
                return charge >= self.source and (charge - self.mod) % self.div == 0
            if self.div == 0:
                return charge >= self.source and charge == self.mod
        return False

    def exec(self, charge):
        return charge - self.source

    def __str__(self):
        # Build the condition part of the rule, e.g., "2a+1" if div=2 and mod=1
        rulecond = "{0}a{1}".format(self.div, "+{0}".format(self.mod) if self.mod > 0 else "")

        # Build the transformation part: "source->target"
        # If target > 0, append "!" to indicate a firing rule (sends spike)
        # If target == 0, it's a forgetting rule (consumes charge only)
        ruletransf = "{0}->{1}{2}".format(self.source, self.target, "!" if self.target > 0 else "")

        # Final string format: condition;transformation;delay
        return "{0};{1};{2}".format(rulecond, ruletransf, self.delay)

class History:
    """
    responsible for recording each neuron status at each step
    takes of advantage of neuron ids starting from 0
    """
    def __init__(self, neurons):
        self.ticks = [] # every element is the state of a neuron
        self.n_len = len(neurons)

        self.add_new_tick() # add initial charge
        for neuron in neurons:
            self.ticks[-1][neuron.nid] = neuron.charge

    def add_new_tick(self):
        self.ticks.append([""] * self.n_len)

    def record_rule(self, neuron, used_rule):
        self.ticks[-1][neuron.nid] = "r:{0}\nc:{1}".format(str(used_rule) if used_rule else "-", neuron.charge)

    def record_incoming(self, neuron, charge, source_nid):
        self.ticks[-1][neuron.nid] += "\ni:{0}({2})\nc:{1}".format(charge, neuron.charge, source_nid)

    # Create and populate output table
    def __str__(self):
        ticks_wrapper = []
        for idx, tick in enumerate(self.ticks):
            pref = idx - 1 if idx > 0 else "initial\ncharge"
            ticks_wrapper.append([pref] + tick)

        table = Texttable()
        table.set_cols_width([7] + [20] * len(self.ticks[0]))
        table.header(["Step"] + ["Neuron " + str(nid) for nid in range(len(self.ticks[0]))])
        table.add_rows(ticks_wrapper[:], header=False)

        return str(table.draw())
        