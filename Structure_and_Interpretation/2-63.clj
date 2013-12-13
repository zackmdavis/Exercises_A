(defn entry [tree]
  (first tree))

(defn left [tree]
  (nth tree 1))

(defn right [tree]
  (nth tree 2))

(defn make-tree [entry left right]
  (list entry left right))

(defn element? [e S]
  (cond (empty? S) false
        (= e (entry S)) true
        (< e (entry S)) (element? e (left S))
        (> e (entry S)) (element? e (right S))))

(defn adjoin [e S]
  (cond (empty? S) (make-tree e '() '())
        (= e (entry S)) S
        (< e (entry S)) (make-tree (entry S)
                                   (adjoin e (left S))
                                   (right S))
        (> e (entry S)) (make-tree (entry S)
                                   (left S)
                                   (adjoin e (right S)))))

(defn leaf-tree [e]
  (make-tree e '() '()))

(def my-first-tree
  (adjoin 1
          (adjoin 4
                  (adjoin 9
                          (adjoin 2
                                  (adjoin 4
                                          (adjoin 3
                                                  (leaf-tree 5))))))))

(println my-first-tree)
; => (5 (3 (2 (1 () ()) ()) (4 () ())) (9 () ())) [looks OK]

(defn tree-to-list1 [tree]
  (if (empty? tree)
    '()
    (concat (tree-to-list1 (left tree))
            (cons (entry tree)
                  (tree-to-list1 (right tree))))))

(println (tree-to-list1 my-first-tree)) ; => (1 2 3 4 5 9) [OK]

(defn copy-to-list [tree result]
  (if (empty? tree)
    result
    (copy-to-list (left tree)
                  (cons (entry tree)
                        (copy-to-list (right tree)
                                      result)))))

(defn tree-to-list2 [tree]
  (copy-to-list tree '()))

(println (tree-to-list2 my-first-tree)) ; => (1 2 3 4 5 9) [OK]

(def Atree
  (make-tree 7
             (make-tree 3 
                        (leaf-tree 1)
                        (leaf-tree 5))
             (make-tree 9
                        '()
                        (leaf-tree 11))))

(def Btree
  (make-tree 3
             (leaf-tree 1)
             (make-tree 7
                        (leaf-tree 5)
                        (make-tree 9
                                   '()
                                   (leaf-tree 11)))))

(def Ctree
  (make-tree 5
             (make-tree 3
                        (leaf-tree 1)
                        '())
             (make-tree 9
                        (leaf-tree 7)
                        (leaf-tree 11))))

(println Atree)
(println Btree)
(println Ctree)
; =>
;; (7 (3 (1 () ()) (5 () ())) (9 () (11 () ())))
;; (3 (1 () ()) (7 (5 () ()) (9 () (11 () ()))))
;; (5 (3 (1 () ()) ()) (9 (7 () ()) (11 () ()))) [looks OK]

(println (tree-to-list1 Atree))
(println (tree-to-list1 Btree))
(println (tree-to-list1 Ctree))

(println (tree-to-list2 Atree))
(println (tree-to-list2 Btree))
(println (tree-to-list2 Ctree))

;; Both procedures seem to correctly convert a treeset into its
;; corresponding ordered list representation.