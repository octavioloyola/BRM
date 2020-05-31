# Bagging Random Miner (BRM)

* Training phase:
    * Input:
        * T: training dataset;
        * τ: number of classifiers in the ensemble;
        * μ: size of training dataset to bootstrap.
    * Output:
        * P: the set of classifiers parameters (selected objects and dissimilarity thresholds).
    * Training:
        1. Set P initially empty; i.e., P←{}.
        1. for i=1..τ do
            1. Let T_i contains a sample with a replacement of μ objects from T.
            1. Let δ_i contains the average dissimilarity between all the pairs of instances in T_i.
            1. P←P⋃{(T_i, δ_i )}
        1. return P.
    * Classification phase:
        * Input:
        * x: instance to be classified;
        * P: the set of parameters computed in the training phase.
        * Q: queue with past classification results;
        * σ: number of past objects to consider in the current classification.
    * Output:
        * Anomaly score.
    * Classification:
        1. Let s←0 be the score computed by the classifiers.
        1. for each (T_i, δ_i ) in P do
            1. Let d_min be the dissimilarity between x and its nearest neighbor in T_i.
            1. Update the score as follows s←s+e^(-0.5(d_min∕δ_i )^2 )
            1. Average the score as follows s←s∕|P|
            1. Let s′ be the average of values in Q.
            1. if |Q|=σ then
                1. dequeue(Q)
            1. enqueue(s, Q)
            1. return (s′+s)/2