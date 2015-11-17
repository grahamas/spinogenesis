/*
* Copyright 2014-2015 Friedemann Zenke
* Extended by Graham Smith, 2015
*
* This file is part of Auryn, a simulation package for plastic
* spiking neural networks.
*
* Auryn is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* Auryn is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with Auryn.  If not, see <http://www.gnu.org/licenses/>.
*
* If you are using Auryn or parts of it for your work please cite:
* Zenke, F. and Gerstner, W., 2014. Limits to high-speed simulations
* of spiking neural networks using general-purpose computers.
* Front Neuroinform 8, 76. doi: 10.3389/fninf.2014.00076
*/



#ifndef SPINESTDPWDCONNECTION_H
#define SPINESTDPWDCONNECTION_H


#include "auryn_definitions.h"
#include "DuplexConnection.h"
#include "EulerTrace.h"
#include "LinearTrace.h"
#include "SpikeDelay.h"


using namespace std;


/*! \brief Doublet STDP All-to-All as implemented in NEST as stdp_synapse_hom
 *
 * This class implements a range of doublet STDP rules including weight dependent
 * ones (hence the wd suffix in he classn ame).
 * It is meant to be similar to stdp_synapse_hom in NEST.
 *
 */
class SpineSTDPwdConnection : public DuplexConnection
{

private:
	AurynWeight learning_rate;

	AurynWeight param_lambda;
	AurynWeight param_alpha;

	AurynWeight param_mu_plus;
	AurynWeight param_mu_minus;

    AurynWeight spine_birth_probability;

	void init(AurynWeight lambda, AurynWeight maxweight);
	void init_shortcuts();

    /*! WaitingSynapse is in a linked list a node representing a new synapse to be
    added to the weight matrix. Major and minor allows the nodes to be used for
    both forward and backward propagation matrices. */
	typedef struct WaitingSynapse{
        AurynWeight weight;
        NeuronID major; // post in fwd; pre in bkw
        NeuronID minor;
        WaitingSynapse *next;
        WaitingSynapse *prev;
        WaitingSynapse():weight(NULL),major(NULL),minor(NULL),next(NULL),prev(NULL) {}
	} WaitingSynapse;

    /*! WaitingList is a wrapper linked list with nodes of type WaitingSynapse.
    The list is ordered by WaitingSynapse.major */
	typedef struct WaitingList {
        WaitingSynapse *first;
        WaitingSynapse *last;
        AurynLong n_waiting;
        WaitingList():first(NULL),last(NULL),num_waiting(0) {}
        WaitingSynapse *pop_first() {
            WaitingSynapse *popped_ref = first;
            first = popped_ref->next;
            first->prev = NULL;
            return popped_value;
        }
        void insert_synapse(WaitingSynapse waiting) {
            bool inserted = false;
            for (WaitingSynapse *curr = first; curr != NULL; curr = curr->next)
            {
                if (curr->major <= waiting->major) {
                    waiting->prev = curr->prev;
                    waiting->next = curr;
                    curr->prev->next = curr->prev;
                    curr->prev = waiting;
                    if (first == curr) {
                        first = waiting;
                    }
                    inserted = true;
                    break;
                }
            }
            if (!inserted)
            {
                last->next = waiting;
                waiting->prev = last;
                last = waiting;
            }
            n_waiting++;
        }
        void insert_synapse(NeuronID major, NeuronID minor, AurynWeight weight) {
            WaitingSynapse waiting = new WaitingSynapse();
            waiting.major = major;
            waiting.minor = minor;
            waiting.weight = weight;
            insert_synapse(waiting);
        }
	} WaitingList;

protected:

	AurynWeight tau_plus;
	AurynWeight tau_minus;

	NeuronID * fwd_ind;
	AurynWeight * fwd_data;

	NeuronID * bkw_ind;
	AurynWeight ** bkw_data;

	PRE_TRACE_MODEL * tr_pre;
	DEFAULT_TRACE_MODEL * tr_post;

    WaitingList fwd_waiting;
	WaitingList bkw_waiting;

	AurynWeight fudge_pot;
	AurynWeight fudge_dep;


	void propagate_forward();
	void propagate_backward();

	void compute_fudge_factors();

public:

	bool stdp_active;

	SpineSTDPwdConnection(SpikingGroup * source, NeuronGroup * destination,
			TransmitterType transmitter=GLUT);

	SpineSTDPwdConnection(SpikingGroup * source, NeuronGroup * destination,
			const char * filename,
			AurynWeight lambda=1e-5,
			AurynWeight maxweight=0.1 ,
			TransmitterType transmitter=GLUT);

	SpineSTDPwdConnection(SpikingGroup * source, NeuronGroup * destination,
			AurynWeight weight, AurynWeight sparseness=0.05,
			AurynWeight lambda=0.01,
			AurynWeight maxweight=100. ,
			TransmitterType transmitter=GLUT,
			string name = "SpineSTDPwdConnection" );

	void set_alpha(AurynWeight a);
	void set_lambda(AurynWeight l);

	void set_mu_plus(AurynWeight m);
	void set_mu_minus(AurynWeight m);

	void set_spine_birth_probability(AurynWeight sbp);

	void set_max_weight(AurynWeight w);

	void new_synapse(NeuronID pre, NeuronID post, AurynWeight weight);

	virtual ~SpineSTDPwdConnection();
	virtual void finalize();
	void free();

	virtual void propagate();
	virtual void evolve();

};

#endif // SPINESTDPWDCONNECTION_H
