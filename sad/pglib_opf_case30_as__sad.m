%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%                                                                  %%%%%
%%%%    IEEE PES Power Grid Library - Optimal Power Flow - v23.07     %%%%%
%%%%          (https://github.com/power-grid-lib/pglib-opf)           %%%%%
%%%%             Benchmark Group - Small Angle Difference             %%%%%
%%%%                         23 - July - 2023                         %%%%%
%%%%                                                                  %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function mpc = pglib_opf_case30_as__sad
mpc.version = '2';
mpc.baseMVA = 100.0;

%% area data
%	area	refbus
mpc.areas = [
	1	 1;
];

%% bus data
%	bus_i	type	Pd	Qd	Gs	Bs	area	Vm	Va	baseKV	zone	Vmax	Vmin
mpc.bus = [
	1	 3	 0.0	 0.0	 0.0	 0.0	 1	    1.00000	    0.00000	 135.0	 1	    1.05000	    0.95000;
	2	 2	 21.7	 12.7	 0.0	 0.0	 1	    1.02500	    0.00000	 135.0	 1	    1.10000	    0.95000;
	3	 1	 2.4	 1.2	 0.0	 0.0	 1	    1.00000	    0.00000	 135.0	 1	    1.05000	    0.95000;
	4	 1	 7.6	 1.6	 0.0	 0.0	 1	    1.00000	    0.00000	 135.0	 1	    1.05000	    0.95000;
	5	 1	 94.2	 19.0	 0.0	 0.0	 1	    1.00000	    0.00000	 135.0	 1	    1.05000	    0.95000;
	6	 1	 0.0	 0.0	 0.0	 0.0	 1	    1.00000	    0.00000	 135.0	 1	    1.05000	    0.95000;
	7	 1	 22.8	 10.9	 0.0	 0.0	 1	    1.00000	    0.00000	 135.0	 1	    1.05000	    0.95000;
	8	 1	 30.0	 30.0	 0.0	 0.0	 1	    1.00000	    0.00000	 135.0	 1	    1.05000	    0.95000;
	9	 1	 0.0	 0.0	 0.0	 0.0	 1	    1.00000	    0.00000	 135.0	 1	    1.05000	    0.95000;
	10	 1	 5.8	 2.0	 0.0	 5.26	 1	    1.00000	    0.00000	 135.0	 1	    1.05000	    0.95000;
	11	 1	 0.0	 0.0	 0.0	 0.0	 1	    1.00000	    0.00000	 135.0	 1	    1.05000	    0.95000;
	12	 1	 11.2	 7.5	 0.0	 0.0	 1	    1.00000	    0.00000	 135.0	 1	    1.05000	    0.95000;
	13	 2	 0.0	 0.0	 0.0	 0.0	 1	    1.02500	    0.00000	 135.0	 1	    1.10000	    0.95000;
	14	 1	 6.2	 1.6	 0.0	 0.0	 1	    1.00000	    0.00000	 135.0	 1	    1.05000	    0.95000;
	15	 1	 8.2	 2.5	 0.0	 0.0	 1	    1.00000	    0.00000	 135.0	 1	    1.05000	    0.95000;
	16	 1	 3.5	 1.8	 0.0	 0.0	 1	    1.00000	    0.00000	 135.0	 1	    1.05000	    0.95000;
	17	 1	 9.0	 5.8	 0.0	 0.0	 1	    1.00000	    0.00000	 135.0	 1	    1.05000	    0.95000;
	18	 1	 3.2	 0.9	 0.0	 0.0	 1	    1.00000	    0.00000	 135.0	 1	    1.05000	    0.95000;
	19	 1	 9.5	 3.4	 0.0	 0.0	 1	    1.00000	    0.00000	 135.0	 1	    1.05000	    0.95000;
	20	 1	 2.2	 0.7	 0.0	 0.0	 1	    1.00000	    0.00000	 135.0	 1	    1.05000	    0.95000;
	21	 1	 17.5	 11.2	 0.0	 0.0	 1	    1.00000	    0.00000	 135.0	 1	    1.05000	    0.95000;
	22	 2	 0.0	 0.0	 0.0	 0.0	 1	    1.02500	    0.00000	 135.0	 1	    1.10000	    0.95000;
	23	 2	 3.2	 1.6	 0.0	 0.0	 1	    1.02500	    0.00000	 135.0	 1	    1.10000	    0.95000;
	24	 1	 8.7	 6.7	 0.0	 25.0	 1	    1.00000	    0.00000	 135.0	 1	    1.05000	    0.95000;
	25	 1	 0.0	 0.0	 0.0	 0.0	 1	    1.00000	    0.00000	 135.0	 1	    1.05000	    0.95000;
	26	 1	 3.5	 2.3	 0.0	 0.0	 1	    1.00000	    0.00000	 135.0	 1	    1.05000	    0.95000;
	27	 2	 0.0	 0.0	 0.0	 0.0	 1	    1.02500	    0.00000	 135.0	 1	    1.10000	    0.95000;
	28	 1	 0.0	 0.0	 0.0	 0.0	 1	    1.00000	    0.00000	 135.0	 1	    1.05000	    0.95000;
	29	 1	 2.4	 0.9	 0.0	 0.0	 1	    1.00000	    0.00000	 135.0	 1	    1.05000	    0.95000;
	30	 1	 10.6	 1.9	 0.0	 0.0	 1	    1.00000	    0.00000	 135.0	 1	    1.05000	    0.95000;
];

%% generator data
%	bus	Pg	Qg	Qmax	Qmin	Vg	mBase	status	Pmax	Pmin
mpc.gen = [
	1	 125.0	 115.0	 250.0	 -20.0	 1.0	 100.0	 1	 200.0	 50.0;
	2	 50.0	 40.0	 100.0	 -20.0	 1.025	 100.0	 1	 80.0	 20.0;
	5	 32.5	 32.5	 80.0	 -15.0	 1.0	 100.0	 1	 50.0	 15.0;
	8	 22.5	 22.5	 60.0	 -15.0	 1.0	 100.0	 1	 35.0	 10.0;
	11	 20.0	 20.0	 50.0	 -10.0	 1.0	 100.0	 1	 30.0	 10.0;
	13	 26.0	 22.5	 60.0	 -15.0	 1.025	 100.0	 1	 40.0	 12.0;
];

%% generator cost data
%	2	startup	shutdown	n	c(n-1)	...	c0
mpc.gencost = [
	2	 0.0	 0.0	 3	   0.003750	   2.000000	   0.000000;
	2	 0.0	 0.0	 3	   0.017500	   1.750000	   0.000000;
	2	 0.0	 0.0	 3	   0.062500	   1.000000	   0.000000;
	2	 0.0	 0.0	 3	   0.008340	   3.250000	   0.000000;
	2	 0.0	 0.0	 3	   0.025000	   3.000000	   0.000000;
	2	 0.0	 0.0	 3	   0.025000	   3.000000	   0.000000;
];

%% branch data
%	fbus	tbus	r	x	b	rateA	rateB	rateC	ratio	angle	status	angmin	angmax
mpc.branch = [
	1	 2	 0.0192	 0.0575	 0.0264	 130.0	 130.0	 130.0	 0.0	 0.0	 1	 -3.50098743806	 3.50098743806;
	1	 3	 0.0452	 0.1852	 0.0204	 130.0	 130.0	 130.0	 0.0	 0.0	 1	 -3.50098743806	 3.50098743806;
	2	 4	 0.057	 0.1737	 0.0184	 65.0	 65.0	 65.0	 0.0	 0.0	 1	 -3.50098743806	 3.50098743806;
	3	 4	 0.0132	 0.0379	 0.0042	 130.0	 130.0	 130.0	 0.0	 0.0	 1	 -3.50098743806	 3.50098743806;
	2	 5	 0.0472	 0.1983	 0.0209	 130.0	 130.0	 130.0	 0.0	 0.0	 1	 -3.50098743806	 3.50098743806;
	2	 6	 0.0581	 0.1763	 0.0187	 65.0	 65.0	 65.0	 0.0	 0.0	 1	 -3.50098743806	 3.50098743806;
	4	 6	 0.0119	 0.0414	 0.0045	 90.0	 90.0	 90.0	 0.0	 0.0	 1	 -3.50098743806	 3.50098743806;
	5	 7	 0.046	 0.116	 0.0102	 70.0	 70.0	 70.0	 0.0	 0.0	 1	 -3.50098743806	 3.50098743806;
	6	 7	 0.0267	 0.082	 0.0085	 130.0	 130.0	 130.0	 0.0	 0.0	 1	 -3.50098743806	 3.50098743806;
	6	 8	 0.012	 0.042	 0.0045	 32.0	 32.0	 32.0	 0.0	 0.0	 1	 -3.50098743806	 3.50098743806;
	6	 9	 0.0	 0.208	 0.0	 65.0	 65.0	 65.0	 0.0	 0.0	 1	 -3.50098743806	 3.50098743806;
	6	 10	 0.0	 0.556	 0.0	 32.0	 32.0	 32.0	 0.0	 0.0	 1	 -3.50098743806	 3.50098743806;
	9	 11	 0.0	 0.208	 0.0	 65.0	 65.0	 65.0	 0.0	 0.0	 1	 -3.50098743806	 3.50098743806;
	9	 10	 0.0	 0.11	 0.0	 65.0	 65.0	 65.0	 0.0	 0.0	 1	 -3.50098743806	 3.50098743806;
	4	 12	 0.0	 0.256	 0.0	 65.0	 65.0	 65.0	 0.0	 0.0	 1	 -3.50098743806	 3.50098743806;
	12	 13	 0.0	 0.14	 0.0	 65.0	 65.0	 65.0	 0.0	 0.0	 1	 -3.50098743806	 3.50098743806;
	12	 14	 0.1231	 0.2559	 0.0	 32.0	 32.0	 32.0	 0.0	 0.0	 1	 -3.50098743806	 3.50098743806;
	12	 15	 0.0662	 0.1304	 0.0	 32.0	 32.0	 32.0	 0.0	 0.0	 1	 -3.50098743806	 3.50098743806;
	12	 16	 0.0945	 0.1987	 0.0	 32.0	 32.0	 32.0	 0.0	 0.0	 1	 -3.50098743806	 3.50098743806;
	14	 15	 0.221	 0.1997	 0.0	 16.0	 16.0	 16.0	 0.0	 0.0	 1	 -3.50098743806	 3.50098743806;
	16	 17	 0.0824	 0.1932	 0.0	 16.0	 16.0	 16.0	 0.0	 0.0	 1	 -3.50098743806	 3.50098743806;
	15	 18	 0.107	 0.2185	 0.0	 16.0	 16.0	 16.0	 0.0	 0.0	 1	 -3.50098743806	 3.50098743806;
	18	 19	 0.0639	 0.1292	 0.0	 16.0	 16.0	 16.0	 0.0	 0.0	 1	 -3.50098743806	 3.50098743806;
	19	 20	 0.034	 0.068	 0.0	 32.0	 32.0	 32.0	 0.0	 0.0	 1	 -3.50098743806	 3.50098743806;
	10	 20	 0.0936	 0.209	 0.0	 32.0	 32.0	 32.0	 0.0	 0.0	 1	 -3.50098743806	 3.50098743806;
	10	 17	 0.0324	 0.0845	 0.0	 32.0	 32.0	 32.0	 0.0	 0.0	 1	 -3.50098743806	 3.50098743806;
	10	 21	 0.0348	 0.0749	 0.0	 32.0	 32.0	 32.0	 0.0	 0.0	 1	 -3.50098743806	 3.50098743806;
	10	 22	 0.0727	 0.1499	 0.0	 32.0	 32.0	 32.0	 0.0	 0.0	 1	 -3.50098743806	 3.50098743806;
	21	 22	 0.0116	 0.0236	 0.0	 32.0	 32.0	 32.0	 0.0	 0.0	 1	 -3.50098743806	 3.50098743806;
	15	 23	 0.1	 0.202	 0.0	 16.0	 16.0	 16.0	 0.0	 0.0	 1	 -3.50098743806	 3.50098743806;
	22	 24	 0.115	 0.179	 0.0	 16.0	 16.0	 16.0	 0.0	 0.0	 1	 -3.50098743806	 3.50098743806;
	23	 24	 0.132	 0.27	 0.0	 16.0	 16.0	 16.0	 0.0	 0.0	 1	 -3.50098743806	 3.50098743806;
	24	 25	 0.1885	 0.3292	 0.0	 16.0	 16.0	 16.0	 0.0	 0.0	 1	 -3.50098743806	 3.50098743806;
	25	 26	 0.2544	 0.38	 0.0	 16.0	 16.0	 16.0	 0.0	 0.0	 1	 -3.50098743806	 3.50098743806;
	25	 27	 0.1093	 0.2087	 0.0	 16.0	 16.0	 16.0	 0.0	 0.0	 1	 -3.50098743806	 3.50098743806;
	28	 27	 0.0	 0.396	 0.0	 65.0	 65.0	 65.0	 0.0	 0.0	 1	 -3.50098743806	 3.50098743806;
	27	 29	 0.2198	 0.4153	 0.0	 16.0	 16.0	 16.0	 0.0	 0.0	 1	 -3.50098743806	 3.50098743806;
	27	 30	 0.3202	 0.6027	 0.0	 16.0	 16.0	 16.0	 0.0	 0.0	 1	 -3.50098743806	 3.50098743806;
	29	 30	 0.2399	 0.4533	 0.0	 16.0	 16.0	 16.0	 0.0	 0.0	 1	 -3.50098743806	 3.50098743806;
	8	 28	 0.0636	 0.2	 0.0214	 32.0	 32.0	 32.0	 0.0	 0.0	 1	 -3.50098743806	 3.50098743806;
	6	 28	 0.0169	 0.0599	 0.0065	 32.0	 32.0	 32.0	 0.0	 0.0	 1	 -3.50098743806	 3.50098743806;
];

% INFO    : === Translation Options ===
% INFO    : Phase Angle Bound:           3.50098743806 (deg.)
% INFO    : 
% INFO    : === Generator Bounds Update Notes ===
% INFO    : 
% INFO    : === Base KV Replacement Notes ===
% INFO    : 
% INFO    : === Transformer Setting Replacement Notes ===
% INFO    : 
% INFO    : === Line Capacity Monotonicity Notes ===
% INFO    : 
% INFO    : === Writing Matpower Case File Notes ===
