(set-logic QF_NRA_ODE)
(declare-fun x3 () Real [0.000000, 10.000000])
(declare-fun x2 () Real [0.000000, 10.000000])
(declare-fun x1 () Real [0.000000, 10.000000])
(declare-fun tau () Real [0.000000, 1.000000])
(declare-fun x3_0_0 () Real [0.000000, 10.000000])
(declare-fun x3_0_t () Real [0.000000, 10.000000])
(declare-fun x3_1_0 () Real [0.000000, 10.000000])
(declare-fun x3_1_t () Real [0.000000, 10.000000])
(declare-fun x3_2_0 () Real [0.000000, 10.000000])
(declare-fun x3_2_t () Real [0.000000, 10.000000])
(declare-fun x2_0_0 () Real [0.000000, 10.000000])
(declare-fun x2_0_t () Real [0.000000, 10.000000])
(declare-fun x2_1_0 () Real [0.000000, 10.000000])
(declare-fun x2_1_t () Real [0.000000, 10.000000])
(declare-fun x2_2_0 () Real [0.000000, 10.000000])
(declare-fun x2_2_t () Real [0.000000, 10.000000])
(declare-fun x1_0_0 () Real [0.000000, 10.000000])
(declare-fun x1_0_t () Real [0.000000, 10.000000])
(declare-fun x1_1_0 () Real [0.000000, 10.000000])
(declare-fun x1_1_t () Real [0.000000, 10.000000])
(declare-fun x1_2_0 () Real [0.000000, 10.000000])
(declare-fun x1_2_t () Real [0.000000, 10.000000])
(declare-fun tau_0_0 () Real [0.000000, 1.000000])
(declare-fun tau_0_t () Real [0.000000, 1.000000])
(declare-fun tau_1_0 () Real [0.000000, 1.000000])
(declare-fun tau_1_t () Real [0.000000, 1.000000])
(declare-fun tau_2_0 () Real [0.000000, 1.000000])
(declare-fun tau_2_t () Real [0.000000, 1.000000])
(declare-fun time_0 () Real [0.000000, 1.000000])
(declare-fun time_1 () Real [0.000000, 1.000000])
(declare-fun time_2 () Real [0.000000, 1.000000])
(declare-fun mode_0 () Real [1.000000, 8.000000])
(declare-fun mode_1 () Real [1.000000, 8.000000])
(declare-fun mode_2 () Real [1.000000, 8.000000])
(define-ode flow_1 ((= d/dt[tau] 1) (= d/dt[x1] (/ (- 5 (* (* 0.5 (^ (* 2 9.80665) 0.5)) (^ x1 0.5))) 2)) (= d/dt[x2] (/ (+ 3 (* (* 0.5 (^ (* 2 9.80665) 0.5)) (- (^ x1 0.5) (^ x2 0.5)))) 4)) (= d/dt[x3] (/ (+ 4 (* (* 0.5 (^ (* 2 9.80665) 0.5)) (- (^ x2 0.5) (^ x3 0.5)))) 3))))
(define-ode flow_2 ((= d/dt[tau] 1) (= d/dt[x1] (/ (- 5 (* (* 0.5 (^ (* 2 9.80665) 0.5)) (^ x1 0.5))) 2)) (= d/dt[x2] (/ (+ 3 (* (* 0.5 (^ (* 2 9.80665) 0.5)) (- (^ x1 0.5) (^ x2 0.5)))) 4)) (= d/dt[x3] (/ (* (* 0.5 (^ (* 2 9.80665) 0.5)) (- (^ x2 0.5) (^ x3 0.5))) 3))))
(define-ode flow_3 ((= d/dt[tau] 1) (= d/dt[x1] (/ (- 5 (* (* 0.5 (^ (* 2 9.80665) 0.5)) (^ x1 0.5))) 2)) (= d/dt[x2] (/ (* (* 0.5 (^ (* 2 9.80665) 0.5)) (- (^ x1 0.5) (^ x2 0.5))) 4)) (= d/dt[x3] (/ (+ 4 (* (* 0.5 (^ (* 2 9.80665) 0.5)) (- (^ x2 0.5) (^ x3 0.5)))) 3))))
(define-ode flow_4 ((= d/dt[tau] 1) (= d/dt[x1] (/ (- 5 (* (* 0.5 (^ (* 2 9.80665) 0.5)) (^ x1 0.5))) 2)) (= d/dt[x2] (/ (* (* 0.5 (^ (* 2 9.80665) 0.5)) (- (^ x1 0.5) (^ x2 0.5))) 4)) (= d/dt[x3] (/ (* (* 0.5 (^ (* 2 9.80665) 0.5)) (- (^ x2 0.5) (^ x3 0.5))) 3))))
(define-ode flow_5 ((= d/dt[tau] 1) (= d/dt[x1] (/ (* (* -0.5 (^ (* 2 9.80665) 0.5)) (^ x1 0.5)) 2)) (= d/dt[x2] (/ (+ 3 (* (* 0.5 (^ (* 2 9.80665) 0.5)) (- (^ x1 0.5) (^ x2 0.5)))) 4)) (= d/dt[x3] (/ (+ 4 (* (* 0.5 (^ (* 2 9.80665) 0.5)) (- (^ x2 0.5) (^ x3 0.5)))) 3))))
(define-ode flow_6 ((= d/dt[tau] 1) (= d/dt[x1] (/ (* (* -0.5 (^ (* 2 9.80665) 0.5)) (^ x1 0.5)) 2)) (= d/dt[x2] (/ (+ 3 (* (* 0.5 (^ (* 2 9.80665) 0.5)) (- (^ x1 0.5) (^ x2 0.5)))) 4)) (= d/dt[x3] (/ (* (* 0.5 (^ (* 2 9.80665) 0.5)) (- (^ x2 0.5) (^ x3 0.5))) 3))))
(define-ode flow_7 ((= d/dt[tau] 1) (= d/dt[x1] (/ (* (* -0.5 (^ (* 2 9.80665) 0.5)) (^ x1 0.5)) 2)) (= d/dt[x2] (/ (* (* 0.5 (^ (* 2 9.80665) 0.5)) (- (^ x1 0.5) (^ x2 0.5))) 4)) (= d/dt[x3] (/ (+ 4 (* (* 0.5 (^ (* 2 9.80665) 0.5)) (- (^ x2 0.5) (^ x3 0.5)))) 3))))
(define-ode flow_8 ((= d/dt[tau] 1) (= d/dt[x1] (/ (* (* -0.5 (^ (* 2 9.80665) 0.5)) (^ x1 0.5)) 2)) (= d/dt[x2] (/ (* (* 0.5 (^ (* 2 9.80665) 0.5)) (- (^ x1 0.5) (^ x2 0.5))) 4)) (= d/dt[x3] (/ (* (* 0.5 (^ (* 2 9.80665) 0.5)) (- (^ x2 0.5) (^ x3 0.5))) 3))))
(assert (and (or (and (= mode_2 8) (or (> x3_2_t (+ 5 0.5)) (< x3_2_t (- 5 0.5))) (or (> x2_2_t (+ 5 0.5)) (< x2_2_t (- 5 0.5))) (or (> x1_2_t (+ 5 0.5)) (< x1_2_t (- 5 0.5)))) (and (= mode_2 7) (or (> x3_2_t (+ 5 0.5)) (< x3_2_t (- 5 0.5))) (or (> x2_2_t (+ 5 0.5)) (< x2_2_t (- 5 0.5))) (or (> x1_2_t (+ 5 0.5)) (< x1_2_t (- 5 0.5)))) (and (= mode_2 6) (or (> x3_2_t (+ 5 0.5)) (< x3_2_t (- 5 0.5))) (or (> x2_2_t (+ 5 0.5)) (< x2_2_t (- 5 0.5))) (or (> x1_2_t (+ 5 0.5)) (< x1_2_t (- 5 0.5)))) (and (= mode_2 5) (or (> x3_2_t (+ 5 0.5)) (< x3_2_t (- 5 0.5))) (or (> x2_2_t (+ 5 0.5)) (< x2_2_t (- 5 0.5))) (or (> x1_2_t (+ 5 0.5)) (< x1_2_t (- 5 0.5)))) (and (= mode_2 4) (or (> x3_2_t (+ 5 0.5)) (< x3_2_t (- 5 0.5))) (or (> x2_2_t (+ 5 0.5)) (< x2_2_t (- 5 0.5))) (or (> x1_2_t (+ 5 0.5)) (< x1_2_t (- 5 0.5)))) (and (= mode_2 3) (or (> x3_2_t (+ 5 0.5)) (< x3_2_t (- 5 0.5))) (or (> x2_2_t (+ 5 0.5)) (< x2_2_t (- 5 0.5))) (or (> x1_2_t (+ 5 0.5)) (< x1_2_t (- 5 0.5)))) (and (= mode_2 2) (or (> x3_2_t (+ 5 0.5)) (< x3_2_t (- 5 0.5))) (or (> x2_2_t (+ 5 0.5)) (< x2_2_t (- 5 0.5))) (or (> x1_2_t (+ 5 0.5)) (< x1_2_t (- 5 0.5)))) (and (= mode_2 1) (or (> x3_2_t (+ 5 0.5)) (< x3_2_t (- 5 0.5))) (or (> x2_2_t (+ 5 0.5)) (< x2_2_t (- 5 0.5))) (or (> x1_2_t (+ 5 0.5)) (< x1_2_t (- 5 0.5))))) (and (= tau_0_0 0) (= x3_0_0 5.9849) (= x2_0_0 6.2666) (= x1_0_0 4.7543)) (= mode_0 1) (or (and (= mode_1 8) (>= x3_0_t 5) (>= x2_0_t 5) (>= x1_0_t 5) (= tau_0_t 1) (= tau_1_0 0) (= x3_1_0 x3_0_t) (= x2_1_0 x2_0_t) (= x1_1_0 x1_0_t)) (and (= mode_1 7) (< x3_0_t 5) (>= x2_0_t 5) (>= x1_0_t 5) (= tau_0_t 1) (= tau_1_0 0) (= x3_1_0 x3_0_t) (= x2_1_0 x2_0_t) (= x1_1_0 x1_0_t)) (and (= mode_1 6) (>= x3_0_t 5) (< x2_0_t 5) (>= x1_0_t 5) (= tau_0_t 1) (= tau_1_0 0) (= x3_1_0 x3_0_t) (= x2_1_0 x2_0_t) (= x1_1_0 x1_0_t)) (and (= mode_1 5) (< x3_0_t 5) (< x2_0_t 5) (>= x1_0_t 5) (= tau_0_t 1) (= tau_1_0 0) (= x3_1_0 x3_0_t) (= x2_1_0 x2_0_t) (= x1_1_0 x1_0_t)) (and (= mode_1 4) (>= x3_0_t 5) (>= x2_0_t 5) (< x1_0_t 5) (= tau_0_t 1) (= tau_1_0 0) (= x3_1_0 x3_0_t) (= x2_1_0 x2_0_t) (= x1_1_0 x1_0_t)) (and (= mode_1 3) (< x3_0_t 5) (>= x2_0_t 5) (< x1_0_t 5) (= tau_0_t 1) (= tau_1_0 0) (= x3_1_0 x3_0_t) (= x2_1_0 x2_0_t) (= x1_1_0 x1_0_t)) (and (= mode_1 2) (>= x3_0_t 5) (< x2_0_t 5) (< x1_0_t 5) (= tau_0_t 1) (= tau_1_0 0) (= x3_1_0 x3_0_t) (= x2_1_0 x2_0_t) (= x1_1_0 x1_0_t)) (and (= mode_1 1) (< x3_0_t 5) (< x2_0_t 5) (< x1_0_t 5) (= tau_0_t 1) (= tau_1_0 0) (= x3_1_0 x3_0_t) (= x2_1_0 x2_0_t) (= x1_1_0 x1_0_t))) (= tau_0_t (+ tau_0_0 (* 1 time_0))) (= [tau_0_t x1_0_t x2_0_t x3_0_t] (integral 0. time_0 [tau_0_0 x1_0_0 x2_0_0 x3_0_0] flow_1)) (= mode_0 1) (forall_t 1 [0 time_0] (>= tau_0_t 0)) (>= tau_0_t 0) (>= tau_0_0 0) (forall_t 1 [0 time_0] (<= tau_0_t 1)) (<= tau_0_t 1) (<= tau_0_0 1) (or (and (= mode_2 8) (>= x3_1_t 5) (>= x2_1_t 5) (>= x1_1_t 5) (= tau_1_t 1) (= tau_2_0 0) (= x3_2_0 x3_1_t) (= x2_2_0 x2_1_t) (= x1_2_0 x1_1_t)) (and (= mode_2 7) (< x3_1_t 5) (>= x2_1_t 5) (>= x1_1_t 5) (= tau_1_t 1) (= tau_2_0 0) (= x3_2_0 x3_1_t) (= x2_2_0 x2_1_t) (= x1_2_0 x1_1_t)) (and (= mode_2 6) (>= x3_1_t 5) (< x2_1_t 5) (>= x1_1_t 5) (= tau_1_t 1) (= tau_2_0 0) (= x3_2_0 x3_1_t) (= x2_2_0 x2_1_t) (= x1_2_0 x1_1_t)) (and (= mode_2 5) (< x3_1_t 5) (< x2_1_t 5) (>= x1_1_t 5) (= tau_1_t 1) (= tau_2_0 0) (= x3_2_0 x3_1_t) (= x2_2_0 x2_1_t) (= x1_2_0 x1_1_t)) (and (= mode_2 4) (>= x3_1_t 5) (>= x2_1_t 5) (< x1_1_t 5) (= tau_1_t 1) (= tau_2_0 0) (= x3_2_0 x3_1_t) (= x2_2_0 x2_1_t) (= x1_2_0 x1_1_t)) (and (= mode_2 3) (< x3_1_t 5) (>= x2_1_t 5) (< x1_1_t 5) (= tau_1_t 1) (= tau_2_0 0) (= x3_2_0 x3_1_t) (= x2_2_0 x2_1_t) (= x1_2_0 x1_1_t)) (and (= mode_2 2) (>= x3_1_t 5) (< x2_1_t 5) (< x1_1_t 5) (= tau_1_t 1) (= tau_2_0 0) (= x3_2_0 x3_1_t) (= x2_2_0 x2_1_t) (= x1_2_0 x1_1_t)) (and (= mode_2 1) (< x3_1_t 5) (< x2_1_t 5) (< x1_1_t 5) (= tau_1_t 1) (= tau_2_0 0) (= x3_2_0 x3_1_t) (= x2_2_0 x2_1_t) (= x1_2_0 x1_1_t))) (= tau_1_t (+ tau_1_0 (* 1 time_1))) (= [tau_1_t x1_1_t x2_1_t x3_1_t] (integral 0. time_1 [tau_1_0 x1_1_0 x2_1_0 x3_1_0] flow_6)) (= mode_1 6) (forall_t 6 [0 time_1] (>= tau_1_t 0)) (>= tau_1_t 0) (>= tau_1_0 0) (forall_t 6 [0 time_1] (<= tau_1_t 1)) (<= tau_1_t 1) (<= tau_1_0 1) (= tau_2_t (+ tau_2_0 (* 1 time_2))) (= [tau_2_t x1_2_t x2_2_t x3_2_t] (integral 0. time_2 [tau_2_0 x1_2_0 x2_2_0 x3_2_0] flow_7)) (= mode_2 7) (forall_t 7 [0 time_2] (>= tau_2_t 0)) (>= tau_2_t 0) (>= tau_2_0 0) (forall_t 7 [0 time_2] (<= tau_2_t 1)) (<= tau_2_t 1) (<= tau_2_0 1)))
(check-sat)
(exit)
