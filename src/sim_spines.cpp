/*
* Copyright 2014 Friedemann Zenke
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
*/

#include "auryn.h"
#include "SpineSTDPwdConnection.h"

using namespace std;

namespace po = boost::program_options;
namespace mpi = boost::mpi;

int main(int ac,char *av[]) {
	string dir = ".";

	stringstream oss;
	string strbuf ;
	string msg;

	NeuronID ne = 2000;
//	NeuronID ni = 2000;

	NeuronID nrec = 50;

	double w = 0.1e-3; // 0.1mV PSC size
	double sparseness = 0.1;
	double simtime = 1.;
	double wext = 0.6e-3;

	double lambda = 1e-2;
	// For the benchmark this value was changed to 1e-9 to
	// avoid frequency changes during the simulation. Otherwise
	// lambda should be in the order of 1e-2 - 1e-3.
	double gamma = 6.0;
	double poisson_rate = 20.0e3;

	double sbp = .0001;

	string load = "";
	string save = "";

	string fwmat_ee = "";
//	string fwmat_ei = "";
//	string fwmat_ie = "";
//	string fwmat_ii = "";

	int errcode = 0;



    try {

        po::options_description desc("Allowed options");
        desc.add_options()
            ("help", "produce help message")
            ("simtime", po::value<double>(), "duration of simulation")
            ("sbp", po::value<double>(), "spine birth probability (per timestep)")
            ("gamma", po::value<double>(), "gamma factor for inhibitory weight")
            ("lambda", po::value<double>(), "learning rate")
            ("nu", po::value<double>(), "the external firing rate nu")
            ("dir", po::value<string>(), "dir from file")
            ("load", po::value<string>(), "load from file")
            ("save", po::value<string>(), "save to file")
            ("fee", po::value<string>(), "file with EE connections")
//            ("fei", po::value<string>(), "file with EI connections")
//            ("fie", po::value<string>(), "file with IE connections")
//            ("fii", po::value<string>(), "file with II connections")
        ;

        po::variables_map vm;
        po::store(po::parse_command_line(ac, av, desc), vm);
        po::notify(vm);

        if (vm.count("help")) {
            cout << desc << "\n";
            return 1;
        }

        if (vm.count("simtime")) {
			simtime = vm["simtime"].as<double>();
        }

        if (vm.count("sbp")) {
			sbp = vm["sbp"].as<double>();
        }

        if (vm.count("gamma")) {
			gamma = vm["gamma"].as<double>();
        }

        if (vm.count("lambda")) {
			lambda = vm["lambda"].as<double>();
        }

        if (vm.count("nu")) {
			poisson_rate = vm["nu"].as<double>();
        }

        if (vm.count("dir")) {
			dir = vm["dir"].as<string>();
        }

        if (vm.count("load")) {
			load = vm["load"].as<string>();
        }

        if (vm.count("save")) {
			save = vm["save"].as<string>();
        }

        if (vm.count("fee")) {
			fwmat_ee = vm["fee"].as<string>();
        }

//        if (vm.count("fie")) {
//			fwmat_ie = vm["fie"].as<string>();
//        }
//
//        if (vm.count("fei")) {
//			fwmat_ei = vm["fei"].as<string>();
//        }
//
//        if (vm.count("fii")) {
//			fwmat_ii = vm["fii"].as<string>();
//        }
    }
    catch(exception& e) {
        cerr << "error: " << e.what() << "\n";
        return 1;
    }
    catch(...) {
        cerr << "Exception of unknown type!\n";
    }

	// BEGIN Auryn init
	mpi::environment env(ac, av);
	mpi::communicator world;
	communicator = &world;

	oss << dir  << "/spines." << world.rank() << ".";
	string outputfile = oss.str();

	stringstream logfile;
	logfile << outputfile << "log";
	logger = new Logger(logfile.str(),world.rank(),PROGRESS,EVERYTHING);

	sys = new System(&world);
	// END Auryn init

	logger->msg("Setting up neuron groups ...",PROGRESS,true);
	IafPscDeltaGroup * neurons_e = new IafPscDeltaGroup( ne );
	neurons_e->set_tau_mem(20.0e-3);
	neurons_e->set_tau_ref(2.0e-3);
	neurons_e->e_rest = 0e-3;
	neurons_e->e_reset = 10e-3;
	neurons_e->thr = 20e-3;

//	IafPscDeltaGroup * neurons_i = new IafPscDeltaGroup( ni );
//	neurons_i->set_tau_mem(20.0e-3);
//	neurons_i->set_tau_ref(2.0e-3);
//	neurons_i->e_rest = 0e-3;
//	neurons_i->e_reset = 10e-3;
//	neurons_i->thr = 20e-3;

	logger->msg("Setting up Poisson input ...",PROGRESS,true);
	// The traditional way to implement the network is with
	// independent Poisson noise.
	PoissonStimulator * pstim_e
		= new PoissonStimulator( neurons_e, poisson_rate, wext );
//	PoissonStimulator * pstim_i
//		= new PoissonStimulator( neurons_i, poisson_rate, wext );

	// The following would give correlated poisson noise from a single
	// population of Poisson Neurons.
	// PoissonGroup * poisson
	// 	= new PoissonGroup( ne, poisson_rate );
	// SparseConnection * cone
	// 	= new SparseConnection(poisson,neurons_e, w, sparseness, MEM );
	// SparseConnection * coni
	// 	= new SparseConnection(poisson,neurons_i, w, sparseness, MEM );

	// This would be a solution where independend Poisson spikes
	// are used from two PoissonGroups.
	// PoissonGroup * pstim_e
	// 	= new PoissonGroup( ne, poisson_rate*ne*sparseness );
	// IdentityConnection * ide
	// 	= new IdentityConnection(pstim_e,neurons_e, w, MEM );
	// PoissonGroup * pstim_i
	// 	= new PoissonGroup( ni, poisson_rate*ne*sparseness );
	// IdentityConnection * idi
	// 	= new IdentityConnection(pstim_i,neurons_i, w, MEM );


    logger->msg("Setting up E connections ...",PROGRESS,true);
	SpineSTDPwdConnection * con_ee
		= new SpineSTDPwdConnection(
				neurons_e,
				neurons_e,
				w,
				sparseness
				);
	con_ee->set_transmitter(MEM);
	con_ee->set_name("E->E");
	con_ee->set_max_weight(3*w);
	con_ee->set_alpha(2.02);
	con_ee->set_lambda(lambda);
	logger->msg("Parameter sbp: " + boost::lexical_cast<std::string>(sbp));
	con_ee->set_spine_birth_probability(sbp);

//	SparseConnection * con_ei
//		= new SparseConnection( neurons_e,neurons_i,w,sparseness,MEM);
//
//	logger->msg("Setting up I connections ...",PROGRESS,true);
//	SparseConnection * con_ii
//		= new SparseConnection( neurons_i,neurons_i,-gamma*w,sparseness,MEM);
//	SparseConnection * con_ie
//		= new SparseConnection( neurons_i,neurons_e,-gamma*w,sparseness,MEM);

	msg = "Setting up monitors ...";
	logger->msg(msg,PROGRESS,true);

	stringstream filename;
	filename << outputfile << "e.ras";
	SpikeMonitor * smon_e = new SpikeMonitor( neurons_e, filename.str().c_str(), nrec);

//	filename.str("");
//	filename.clear();
//	filename << outputfile << "i.ras";
//	SpikeMonitor * smon_i = new SpikeMonitor( neurons_i, filename.str().c_str(), nrec);

	// filename.str("");
	// filename.clear();
	// filename << outputfile << "syn";
	// WeightMonitor * wmon = new WeightMonitor( con_ee, filename.str() );
	// wmon->add_equally_spaced(1000);

	// filename.str("");
	// filename.clear();
	// filename << outputfile << "mem";
	// StateMonitor * smon = new StateMonitor( neurons_e, 13, "mem", filename.str() );

	RateChecker * chk = new RateChecker( neurons_e , 0.1 , 1000. , 100e-3);

	if ( !load.empty() ) {
		sys->load_network_state(load);
	}

	if ( !fwmat_ee.empty() ) con_ee->load_from_complete_file(fwmat_ee);
//	if ( !fwmat_ei.empty() ) con_ei->load_from_complete_file(fwmat_ei);
//	if ( !fwmat_ie.empty() ) con_ie->load_from_complete_file(fwmat_ie);
//	if ( !fwmat_ii.empty() ) con_ii->load_from_complete_file(fwmat_ii);

	// con_ee->prune();
	// con_ei->prune();
	// con_ie->prune();
	// con_ii->prune();

	// logger->msg("Running sanity check ...",PROGRESS,true);
	con_ee->sanity_check();
//	con_ei->sanity_check();
//	con_ie->sanity_check();
//	con_ii->sanity_check();

    logger->msg("Fill level: " + boost::lexical_cast<std::string>(con_ee->w->get_fill_level()));

    logger->msg("HI THERE");

	logger->msg("Simulating ..." ,PROGRESS,true);
	logger->msg("simtime: " + boost::lexical_cast<std::string>(simtime));
	if (!sys->run(simtime,true))
			errcode = 1;

	if ( !save.empty() ) {
		sys->save_network_state(save);
	}


	logger->msg("Freeing ..." ,PROGRESS,true);
	delete sys;

	if (errcode)
		env.abort(errcode);

	return errcode;
}
