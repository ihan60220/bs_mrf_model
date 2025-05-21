#include <bits/stdc++.h>

/*
    A 1D MRF model with performance in mind
    All of the arrays will be serialized and turned into numpy arrays, since
    python will be used for all high level operations

*/

class mrf_node {
    private:
        // able to model I(E, kx, ky), the ARPES intensity distribution
        double intensity;
        double E;
        double k;

        // needs to contain the neighbors
        mrf_node* left;
        mrf_node* right;


    public:
        mrf_node() {
            // default constructor
        }

        mrf_node(double intensity, double E, double k) {
            // assignment call
            intensity = intensity;
            E = E;
            k = k;
        }


};


class mrf {
    // needs to contain a 1d-chain of nodes and an associated joint probability distribution
    // functions are not first class objects, so cannot be a member
    
    // since fixed size, might as well make it an array
    private:
        mrf_node* mrf_chain;

    public:
        mrf() {

        }

        mrf(double **intensity, double E[], double k[]) {
            // initialize the mrf with the corresponding I(E, k) distribution
            // intensity is a 2D distribution

            int height, length; // the intensity object will have these associated dimensions

            for (int i = 0; i < height; i++) {
                for (int j = 0; j < length; j++) {
                    // for node (i, j), the associated (E, K) values are (E[j], k[i])
                    
                }
            } 
        }

        void optimize() {
            // the iterated conditional mode optimizer for the mrf
        }

};


int main(void) {

    // create a markov random field object



    return 0;
}