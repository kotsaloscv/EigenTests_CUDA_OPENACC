/*********************************************************
Model Name      : hcn2
Filename        : hcn2.mod
NMODL Version   : 6.2.0
Vectorized      : true
Threadsafe      : true
Created         : Thu May 27 10:01:35 2021
Backend         : C-OpenAcc (api-compatibility)
NMODL Compiler  : 0.2 []
*********************************************************/

#include <math.h>
#include "nmodl/fast_math.hpp" // extend math with some useful functions
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <openacc.h>

#include <coreneuron/mechanism/mech/cfile/scoplib.h>
#include <coreneuron/nrnconf.h>
#include <coreneuron/sim/multicore.hpp>
#include <coreneuron/mechanism/register_mech.hpp>
#include <coreneuron/gpu/nrn_acc_manager.hpp>
#include <coreneuron/utils/randoms/nrnran123.h>
#include <coreneuron/nrniv/nrniv_decl.h>
#include <coreneuron/utils/ivocvect.hpp>
#include <coreneuron/utils/nrnoc_aux.hpp>
#include <coreneuron/mechanism/mech/mod2c_core_thread.hpp>
#include <coreneuron/sim/scopmath/newton_struct.h>
#include "_kinderiv.h"
#include <Eigen/LU>


namespace coreneuron {


    /** channel information */
    static const char *mechanism[] = {
        "6.2.0",
        "hcn2",
        "gbar_hcn2",
        "ehcn_hcn2",
        0,
        "g_hcn2",
        "i_hcn2",
        0,
        "c_hcn2",
        "cac_hcn2",
        "o_hcn2",
        "cao_hcn2",
        0,
        0
    };


    /** all global variables */
    struct hcn2_Store {
        int a_type;
        double c0;
        double cac0;
        double o0;
        double cao0;
        int reset;
        int mech_type;
        double a0;
        double b0;
        double ah;
        double bh;
        double ac;
        double bc;
        double aa0;
        double ba0;
        double aah;
        double bah;
        double aac;
        double bac;
        double kon;
        double koff;
        double b;
        double bf;
        double gca;
        double shift;
        double q10v;
        double q10a;
        int* slist1;
        int* dlist1;
        ThreadDatum* ext_call_thread;
    };

    /** holds object of global variable */
    hcn2_Store hcn2_global;
    #pragma acc declare create (hcn2_global)


    /** all mechanism instance variables */
    struct hcn2_Instance  {
        const double* __restrict__ gbar;
        const double* __restrict__ ehcn;
        double* __restrict__ g;
        double* __restrict__ i;
        double* __restrict__ c;
        double* __restrict__ cac;
        double* __restrict__ o;
        double* __restrict__ cao;
        double* __restrict__ ai;
        double* __restrict__ alpha;
        double* __restrict__ beta;
        double* __restrict__ alphaa;
        double* __restrict__ betaa;
        double* __restrict__ Dc;
        double* __restrict__ Dcac;
        double* __restrict__ Do;
        double* __restrict__ Dcao;
        double* __restrict__ v_unused;
        double* __restrict__ g_unused;
        const double* __restrict__ ion_ai;
    };


    /** connect global (scalar) variables to hoc -- */
    static DoubScal hoc_scalar_double[] = {
        "a0_hcn2", &hcn2_global.a0,
        "b0_hcn2", &hcn2_global.b0,
        "ah_hcn2", &hcn2_global.ah,
        "bh_hcn2", &hcn2_global.bh,
        "ac_hcn2", &hcn2_global.ac,
        "bc_hcn2", &hcn2_global.bc,
        "aa0_hcn2", &hcn2_global.aa0,
        "ba0_hcn2", &hcn2_global.ba0,
        "aah_hcn2", &hcn2_global.aah,
        "bah_hcn2", &hcn2_global.bah,
        "aac_hcn2", &hcn2_global.aac,
        "bac_hcn2", &hcn2_global.bac,
        "kon_hcn2", &hcn2_global.kon,
        "koff_hcn2", &hcn2_global.koff,
        "b_hcn2", &hcn2_global.b,
        "bf_hcn2", &hcn2_global.bf,
        "gca_hcn2", &hcn2_global.gca,
        "shift_hcn2", &hcn2_global.shift,
        "q10v_hcn2", &hcn2_global.q10v,
        "q10a_hcn2", &hcn2_global.q10a,
        0, 0
    };


    /** connect global (array) variables to hoc -- */
    static DoubVec hoc_vector_double[] = {
        0, 0, 0
    };


    static inline int first_pointer_var_index() {
        return -1;
    }


    static inline int float_variables_size() {
        return 19;
    }


    static inline int int_variables_size() {
        return 1;
    }


    static inline int get_mech_type() {
        return hcn2_global.mech_type;
    }


    static inline Memb_list* get_memb_list(NrnThread* nt) {
        if (nt->_ml_list == NULL) {
            return NULL;
        }
        return nt->_ml_list[get_mech_type()];
    }


    static inline void* mem_alloc(size_t num, size_t size, size_t alignment = 16) {
        void* ptr;
        cudaMallocManaged(&ptr, num*size);
        cudaMemset(ptr, 0, num*size);
        return ptr;
    }


    static inline void mem_free(void* ptr) {
        cudaFree(ptr);
    }


    static inline void coreneuron_abort() {
        printf("Error : Issue while running OpenACC kernel \n");
        assert(0==1);
    }


    /** initialize global variables */
    static inline void setup_global_variables()  {
        static int setup_done = 0;
        if (setup_done) {
            return;
        }
        hcn2_global.slist1 = (int*) mem_alloc(4, sizeof(int));
        hcn2_global.dlist1 = (int*) mem_alloc(4, sizeof(int));
        hcn2_global.c0 = 0.0;
        hcn2_global.cac0 = 0.0;
        hcn2_global.o0 = 0.0;
        hcn2_global.cao0 = 0.0;
        hcn2_global.a0 = 0.0015;
        hcn2_global.b0 = 0.02;
        hcn2_global.ah = -135.7;
        hcn2_global.bh = -99.7;
        hcn2_global.ac = -0.155;
        hcn2_global.bc = 0.144;
        hcn2_global.aa0 = 0.0067;
        hcn2_global.ba0 = 0.014;
        hcn2_global.aah = -142.28;
        hcn2_global.bah = -83.5;
        hcn2_global.aac = -0.075;
        hcn2_global.bac = 0.144;
        hcn2_global.kon = 3085.7;
        hcn2_global.koff = 4.4857e-05;
        hcn2_global.b = 80;
        hcn2_global.bf = 8.94;
        hcn2_global.gca = 1;
        hcn2_global.shift = 0;
        hcn2_global.q10v = 4;
        hcn2_global.q10a = 1.5;
        #pragma acc update device (hcn2_global)

        setup_done = 1;
    }


    /** free global variables */
    static inline void free_global_variables()  {
        mem_free(hcn2_global.slist1);
        mem_free(hcn2_global.dlist1);
    }


    /** initialize mechanism instance variables */
    static inline void setup_instance(NrnThread* nt, Memb_list* ml)  {
        hcn2_Instance* inst = (hcn2_Instance*) mem_alloc(1, sizeof(hcn2_Instance));
        int pnodecount = ml->_nodecount_padded;
        Datum* indexes = ml->pdata;
        inst->gbar = (double*) acc_deviceptr(ml->data+0*pnodecount);
        inst->ehcn = (double*) acc_deviceptr(ml->data+1*pnodecount);
        inst->g = (double*) acc_deviceptr(ml->data+2*pnodecount);
        inst->i = (double*) acc_deviceptr(ml->data+3*pnodecount);
        inst->c = (double*) acc_deviceptr(ml->data+4*pnodecount);
        inst->cac = (double*) acc_deviceptr(ml->data+5*pnodecount);
        inst->o = (double*) acc_deviceptr(ml->data+6*pnodecount);
        inst->cao = (double*) acc_deviceptr(ml->data+7*pnodecount);
        inst->ai = (double*) acc_deviceptr(ml->data+8*pnodecount);
        inst->alpha = (double*) acc_deviceptr(ml->data+9*pnodecount);
        inst->beta = (double*) acc_deviceptr(ml->data+10*pnodecount);
        inst->alphaa = (double*) acc_deviceptr(ml->data+11*pnodecount);
        inst->betaa = (double*) acc_deviceptr(ml->data+12*pnodecount);
        inst->Dc = (double*) acc_deviceptr(ml->data+13*pnodecount);
        inst->Dcac = (double*) acc_deviceptr(ml->data+14*pnodecount);
        inst->Do = (double*) acc_deviceptr(ml->data+15*pnodecount);
        inst->Dcao = (double*) acc_deviceptr(ml->data+16*pnodecount);
        inst->v_unused = (double*) acc_deviceptr(ml->data+17*pnodecount);
        inst->g_unused = (double*) acc_deviceptr(ml->data+18*pnodecount);
        inst->ion_ai = (double*) acc_deviceptr(nt->_data);
        ml->instance = (void*) inst;
    }


    /** cleanup mechanism instance variables */
    static inline void cleanup_instance(Memb_list* ml)  {
        hcn2_Instance* inst = (hcn2_Instance*) ml->instance;
        mem_free((void*)inst);
    }


    static void nrn_alloc_hcn2(double* data, Datum* indexes, int type)  {
        // do nothing
    }


    void nrn_destructor_hcn2(NrnThread* nt, Memb_list* ml, int type) {
        int nodecount = ml->nodecount;
        int pnodecount = ml->_nodecount_padded;
        const int* __restrict__ node_index = ml->nodeindices;
        double* __restrict__ data = ml->data;
        const double* __restrict__ voltage = nt->_actual_v;
        Datum* __restrict__ indexes = ml->pdata;
        ThreadDatum* __restrict__ thread = ml->_thread;
        hcn2_Instance* __restrict__ inst = (hcn2_Instance*) ml->instance;

    }


    inline int rates_hcn2(int id, int pnodecount, hcn2_Instance* inst, double* data, const Datum* indexes, ThreadDatum* thread, NrnThread* nt, double v, double arg_v);


    inline int rates_hcn2(int id, int pnodecount, hcn2_Instance* inst, double* data, const Datum* indexes, ThreadDatum* thread, NrnThread* nt, double v, double arg_v) {
        int ret_rates = 0;
        double qv;
        qv = pow(hcn2_global.q10v, ((celsius - 22.0) / 10.0));
        inst->alpha[id] = hcn2_global.a0 * qv / (1.0 + exp( -(arg_v - hcn2_global.ah - hcn2_global.shift) * hcn2_global.ac));
        inst->beta[id] = hcn2_global.b0 * qv / (1.0 + exp( -(arg_v - hcn2_global.bh - hcn2_global.shift) * hcn2_global.bc));
        inst->alphaa[id] = hcn2_global.aa0 * qv / (1.0 + exp( -(arg_v - hcn2_global.aah - hcn2_global.shift) * hcn2_global.aac));
        inst->betaa[id] = hcn2_global.ba0 * qv / (1.0 + exp( -(arg_v - hcn2_global.bah - hcn2_global.shift) * hcn2_global.bac));
        return ret_rates;
    }


    /** initialize channel */
    void nrn_init_hcn2(NrnThread* nt, Memb_list* ml, int type) {
        #pragma acc data present(nt, ml, hcn2_global)
        {
            int nodecount = ml->nodecount;
            int pnodecount = ml->_nodecount_padded;
            const int* __restrict__ node_index = ml->nodeindices;
            double* __restrict__ data = ml->data;
            const double* __restrict__ voltage = nt->_actual_v;
            Datum* __restrict__ indexes = ml->pdata;
            ThreadDatum* __restrict__ thread = ml->_thread;

            setup_instance(nt, ml);
            hcn2_Instance* __restrict__ inst = (hcn2_Instance*) ml->instance;

            if (_nrn_skip_initmodel == 0) {
                int start = 0;
                int end = nodecount;
                #pragma acc parallel loop present(inst, node_index, data, voltage, indexes, thread) async(nt->stream_id)
                for (int id = start; id < end; id++) {
                    int node_id = node_index[id];
                    double v = voltage[node_id];
                    inst->ai[id] = inst->ion_ai[indexes[0*pnodecount + id]];
                    inst->c[id] = hcn2_global.c0;
                    inst->cac[id] = hcn2_global.cac0;
                    inst->o[id] = hcn2_global.o0;
                    inst->cao[id] = hcn2_global.cao0;
                                        
                    Eigen::Matrix<double, 4, 1> X, F;
                    Eigen::Matrix<double, 4, 4> Jm;
                    double* J = Jm.data();
                    double qa, dt_saved_value, old_c, old_cac, old_o;
                    dt_saved_value = nt->_dt;
                    nt->_dt = 1000000000.0;
                    qa = pow(hcn2_global.q10a, ((celsius - 22.0) / 10.0));
                    {
                        double qv, v_in_1;
                        v_in_1 = v;
                        qv = pow(hcn2_global.q10v, ((celsius - 22.0) / 10.0));
                        inst->alpha[id] = hcn2_global.a0 * qv / (1.0 + exp( -(v_in_1 - hcn2_global.ah - hcn2_global.shift) * hcn2_global.ac));
                        inst->beta[id] = hcn2_global.b0 * qv / (1.0 + exp( -(v_in_1 - hcn2_global.bh - hcn2_global.shift) * hcn2_global.bc));
                        inst->alphaa[id] = hcn2_global.aa0 * qv / (1.0 + exp( -(v_in_1 - hcn2_global.aah - hcn2_global.shift) * hcn2_global.aac));
                        inst->betaa[id] = hcn2_global.ba0 * qv / (1.0 + exp( -(v_in_1 - hcn2_global.bah - hcn2_global.shift) * hcn2_global.bac));
                    }
                    old_c = inst->c[id];
                    old_cac = inst->cac[id];
                    old_o = inst->o[id];
                    X[0] = inst->c[id];
                    X[1] = inst->cac[id];
                    X[2] = inst->o[id];
                    X[3] = inst->cao[id];
                    F[0] =  -old_c;
                    J[0] =  -inst->ai[id] * nt->_dt * hcn2_global.kon * qa / hcn2_global.bf - inst->alpha[id] * nt->_dt - 1.0;
                    J[4] = hcn2_global.b * nt->_dt * hcn2_global.koff * qa / hcn2_global.bf;
                    J[8] = inst->beta[id] * nt->_dt;
                    J[12] = 0.0;
                    F[1] =  -old_cac;
                    J[1] = inst->ai[id] * nt->_dt * hcn2_global.kon * qa / hcn2_global.bf;
                    J[5] =  -inst->alphaa[id] * nt->_dt - hcn2_global.b * nt->_dt * hcn2_global.koff * qa / hcn2_global.bf - 1.0;
                    J[9] = 0.0;
                    J[13] = inst->betaa[id] * nt->_dt;
                    F[2] =  -old_o;
                    J[2] = inst->alpha[id] * nt->_dt;
                    J[6] = 0.0;
                    J[10] =  -inst->ai[id] * nt->_dt * hcn2_global.kon * qa - inst->beta[id] * nt->_dt - 1.0;
                    J[14] = nt->_dt * hcn2_global.koff * qa;
                    F[3] =  -1.0;
                    J[3] =  -1.0;
                    J[7] =  -1.0;
                    J[11] =  -1.0;
                    J[15] =  -1.0;

                    X = Jm.inverse()*F;//Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<double, 4, 4>>>(Jm).solve(F);
                    inst->c[id] = X[0];
                    inst->cac[id] = X[1];
                    inst->o[id] = X[2];
                    inst->cao[id] = X[3];
                    nt->_dt = dt_saved_value;


                }
            }
        }
    }


    static inline double nrn_current(int id, int pnodecount, hcn2_Instance* inst, double* data, const Datum* indexes, ThreadDatum* thread, NrnThread* nt, double v) {
        double current = 0.0;
        inst->g[id] = inst->gbar[id] * (inst->o[id] + inst->cao[id] * hcn2_global.gca);
        inst->i[id] = inst->g[id] * (v - inst->ehcn[id]) * (1e-3);
        current += inst->i[id];
        return current;
    }


    /** update current */
    void nrn_cur_hcn2(NrnThread* nt, Memb_list* ml, int type) {
        #pragma acc data present(nt, ml, hcn2_global)
        {
            int nodecount = ml->nodecount;
            int pnodecount = ml->_nodecount_padded;
            const int* __restrict__ node_index = ml->nodeindices;
            double* __restrict__ data = ml->data;
            const double* __restrict__ voltage = nt->_actual_v;
            double* __restrict__  vec_rhs = nt->_actual_rhs;
            double* __restrict__  vec_d = nt->_actual_d;
            Datum* __restrict__ indexes = ml->pdata;
            ThreadDatum* __restrict__ thread = ml->_thread;
            hcn2_Instance* __restrict__ inst = (hcn2_Instance*) ml->instance;

            int start = 0;
            int end = nodecount;
            #pragma acc parallel loop present(inst, node_index, data, voltage, indexes, thread, vec_rhs, vec_d) async(nt->stream_id)
            for (int id = start; id < end; id++) {
                int node_id = node_index[id];
                double v = voltage[node_id];
                inst->ai[id] = inst->ion_ai[indexes[0*pnodecount + id]];
                double g = nrn_current(id, pnodecount, inst, data, indexes, thread, nt, v+0.001);
                double rhs = nrn_current(id, pnodecount, inst, data, indexes, thread, nt, v);
                g = (g-rhs)/0.001;
                #pragma acc atomic update
                vec_rhs[node_id] -= rhs;
                #pragma acc atomic update
                vec_d[node_id] += g;
            }
        }
    }


    /** update state */
    void nrn_state_hcn2(NrnThread* nt, Memb_list* ml, int type) {
        #pragma acc data present(nt, ml, hcn2_global)
        {
            int nodecount = ml->nodecount;
            int pnodecount = ml->_nodecount_padded;
            const int* __restrict__ node_index = ml->nodeindices;
            double* __restrict__ data = ml->data;
            const double* __restrict__ voltage = nt->_actual_v;
            Datum* __restrict__ indexes = ml->pdata;
            ThreadDatum* __restrict__ thread = ml->_thread;
            hcn2_Instance* __restrict__ inst = (hcn2_Instance*) ml->instance;

            int start = 0;
            int end = nodecount;
            #pragma acc parallel loop present(inst, node_index, data, voltage, indexes, thread) async(nt->stream_id)
            for (int id = start; id < end; id++) {
                int node_id = node_index[id];
                double v = voltage[node_id];
                inst->ai[id] = inst->ion_ai[indexes[0*pnodecount + id]];
                
                Eigen::Matrix<double, 4, 1> X, F;
                Eigen::Matrix<double, 4, 4> Jm;
                double* J = Jm.data();
                double qa, old_c, old_cac, old_o;
                qa = pow(hcn2_global.q10a, ((celsius - 22.0) / 10.0));
                {
                    double qv, v_in_0;
                    v_in_0 = v;
                    qv = pow(hcn2_global.q10v, ((celsius - 22.0) / 10.0));
                    inst->alpha[id] = hcn2_global.a0 * qv / (1.0 + exp( -(v_in_0 - hcn2_global.ah - hcn2_global.shift) * hcn2_global.ac));
                    inst->beta[id] = hcn2_global.b0 * qv / (1.0 + exp( -(v_in_0 - hcn2_global.bh - hcn2_global.shift) * hcn2_global.bc));
                    inst->alphaa[id] = hcn2_global.aa0 * qv / (1.0 + exp( -(v_in_0 - hcn2_global.aah - hcn2_global.shift) * hcn2_global.aac));
                    inst->betaa[id] = hcn2_global.ba0 * qv / (1.0 + exp( -(v_in_0 - hcn2_global.bah - hcn2_global.shift) * hcn2_global.bac));
                }
                old_c = inst->c[id];
                old_cac = inst->cac[id];
                old_o = inst->o[id];
                X[0] = inst->c[id];
                X[1] = inst->cac[id];
                X[2] = inst->o[id];
                X[3] = inst->cao[id];
                F[0] =  -old_c;
                J[0] =  -inst->ai[id] * nt->_dt * hcn2_global.kon * qa / hcn2_global.bf - inst->alpha[id] * nt->_dt - 1.0;
                J[4] = hcn2_global.b * nt->_dt * hcn2_global.koff * qa / hcn2_global.bf;
                J[8] = inst->beta[id] * nt->_dt;
                J[12] = 0.0;
                F[1] =  -old_cac;
                J[1] = inst->ai[id] * nt->_dt * hcn2_global.kon * qa / hcn2_global.bf;
                J[5] =  -inst->alphaa[id] * nt->_dt - hcn2_global.b * nt->_dt * hcn2_global.koff * qa / hcn2_global.bf - 1.0;
                J[9] = 0.0;
                J[13] = inst->betaa[id] * nt->_dt;
                F[2] =  -old_o;
                J[2] = inst->alpha[id] * nt->_dt;
                J[6] = 0.0;
                J[10] =  -inst->ai[id] * nt->_dt * hcn2_global.kon * qa - inst->beta[id] * nt->_dt - 1.0;
                J[14] = nt->_dt * hcn2_global.koff * qa;
                F[3] =  -1.0;
                J[3] =  -1.0;
                J[7] =  -1.0;
                J[11] =  -1.0;
                J[15] =  -1.0;

                X = Jm.inverse()*F;//Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<double, 4, 4>>>(Jm).solve(F);
                inst->c[id] = X[0];
                inst->cac[id] = X[1];
                inst->o[id] = X[2];
                inst->cao[id] = X[3];

            }
        }
    }


    /** register channel with the simulator */
    void _hcn2_reg()  {

        int mech_type = nrn_get_mechtype("hcn2");
        hcn2_global.mech_type = mech_type;
        if (mech_type == -1) {
            return;
        }

        _nrn_layout_reg(mech_type, 0);
        register_mech(mechanism, nrn_alloc_hcn2, nrn_cur_hcn2, NULL, nrn_state_hcn2, nrn_init_hcn2, first_pointer_var_index(), 1);
        hcn2_global.a_type = nrn_get_mechtype("a_ion");

        setup_global_variables();
        hoc_register_prop_size(mech_type, float_variables_size(), int_variables_size());
        hoc_register_dparam_semantics(mech_type, 0, "a_ion");
        hoc_register_var(hoc_scalar_double, hoc_vector_double, NULL);
    }
}
