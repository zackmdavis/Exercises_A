(import [3_constraints [*]])

(def s1 (Connector))
(Probe "first operand" s1)

(def s2 (Connector))
(Probe "second operand" s2)

(def output (Connector))

(def the-adder (Adder s1 s2 output))
(print "MY DEBUG MARKER A" (dir the-adder))

(Constant 2 s1)

(Constant 3 s2)
