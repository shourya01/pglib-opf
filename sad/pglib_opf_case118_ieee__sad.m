%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%                                                                  %%%%%
%%%%    IEEE PES Power Grid Library - Optimal Power Flow - v21.07     %%%%%
%%%%          (https://github.com/power-grid-lib/pglib-opf)           %%%%%
%%%%             Benchmark Group - Small Angle Difference             %%%%%
%%%%                         29 - July - 2021                         %%%%%
%%%%                                                                  %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function mpc = pglib_opf_case118_ieee__sad
mpc.version = '2';
mpc.baseMVA = 100.0;

%% bus data
%	bus_i	type	Pd	Qd	Gs	Bs	area	Vm	Va	baseKV	zone	Vmax	Vmin
mpc.bus = [
	1	 2	 51.0	 27.0	 0.0	 0.0	 1	    1.00000	    0.00000	 138.0	 1	    1.06000	    0.94000;
	2	 1	 20.0	 9.0	 0.0	 0.0	 1	    1.00000	    0.00000	 138.0	 1	    1.06000	    0.94000;
	3	 1	 39.0	 10.0	 0.0	 0.0	 1	    1.00000	    0.00000	 138.0	 1	    1.06000	    0.94000;
	4	 2	 39.0	 12.0	 0.0	 0.0	 1	    1.00000	    0.00000	 138.0	 1	    1.06000	    0.94000;
	5	 1	 0.0	 0.0	 0.0	 -40.0	 1	    1.00000	    0.00000	 138.0	 1	    1.06000	    0.94000;
	6	 2	 52.0	 22.0	 0.0	 0.0	 1	    1.00000	    0.00000	 138.0	 1	    1.06000	    0.94000;
	7	 1	 19.0	 2.0	 0.0	 0.0	 1	    1.00000	    0.00000	 138.0	 1	    1.06000	    0.94000;
	8	 2	 28.0	 0.0	 0.0	 0.0	 1	    1.00000	    0.00000	 345.0	 1	    1.06000	    0.94000;
	9	 1	 0.0	 0.0	 0.0	 0.0	 1	    1.00000	    0.00000	 345.0	 1	    1.06000	    0.94000;
	10	 2	 0.0	 0.0	 0.0	 0.0	 1	    1.00000	    0.00000	 345.0	 1	    1.06000	    0.94000;
	11	 1	 70.0	 23.0	 0.0	 0.0	 1	    1.00000	    0.00000	 138.0	 1	    1.06000	    0.94000;
	12	 2	 47.0	 10.0	 0.0	 0.0	 1	    1.00000	    0.00000	 138.0	 1	    1.06000	    0.94000;
	13	 1	 34.0	 16.0	 0.0	 0.0	 1	    1.00000	    0.00000	 138.0	 1	    1.06000	    0.94000;
	14	 1	 14.0	 1.0	 0.0	 0.0	 1	    1.00000	    0.00000	 138.0	 1	    1.06000	    0.94000;
	15	 2	 90.0	 30.0	 0.0	 0.0	 1	    1.00000	    0.00000	 138.0	 1	    1.06000	    0.94000;
	16	 1	 25.0	 10.0	 0.0	 0.0	 1	    1.00000	    0.00000	 138.0	 1	    1.06000	    0.94000;
	17	 1	 11.0	 3.0	 0.0	 0.0	 1	    1.00000	    0.00000	 138.0	 1	    1.06000	    0.94000;
	18	 2	 60.0	 34.0	 0.0	 0.0	 1	    1.00000	    0.00000	 138.0	 1	    1.06000	    0.94000;
	19	 2	 45.0	 25.0	 0.0	 0.0	 1	    1.00000	    0.00000	 138.0	 1	    1.06000	    0.94000;
	20	 1	 18.0	 3.0	 0.0	 0.0	 1	    1.00000	    0.00000	 138.0	 1	    1.06000	    0.94000;
	21	 1	 14.0	 8.0	 0.0	 0.0	 1	    1.00000	    0.00000	 138.0	 1	    1.06000	    0.94000;
	22	 1	 10.0	 5.0	 0.0	 0.0	 1	    1.00000	    0.00000	 138.0	 1	    1.06000	    0.94000;
	23	 1	 7.0	 3.0	 0.0	 0.0	 1	    1.00000	    0.00000	 138.0	 1	    1.06000	    0.94000;
	24	 2	 13.0	 0.0	 0.0	 0.0	 1	    1.00000	    0.00000	 138.0	 1	    1.06000	    0.94000;
	25	 2	 0.0	 0.0	 0.0	 0.0	 1	    1.00000	    0.00000	 138.0	 1	    1.06000	    0.94000;
	26	 2	 0.0	 0.0	 0.0	 0.0	 1	    1.00000	    0.00000	 345.0	 1	    1.06000	    0.94000;
	27	 2	 71.0	 13.0	 0.0	 0.0	 1	    1.00000	    0.00000	 138.0	 1	    1.06000	    0.94000;
	28	 1	 17.0	 7.0	 0.0	 0.0	 1	    1.00000	    0.00000	 138.0	 1	    1.06000	    0.94000;
	29	 1	 24.0	 4.0	 0.0	 0.0	 1	    1.00000	    0.00000	 138.0	 1	    1.06000	    0.94000;
	30	 1	 0.0	 0.0	 0.0	 0.0	 1	    1.00000	    0.00000	 345.0	 1	    1.06000	    0.94000;
	31	 2	 43.0	 27.0	 0.0	 0.0	 1	    1.00000	    0.00000	 138.0	 1	    1.06000	    0.94000;
	32	 2	 59.0	 23.0	 0.0	 0.0	 1	    1.00000	    0.00000	 138.0	 1	    1.06000	    0.94000;
	33	 1	 23.0	 9.0	 0.0	 0.0	 1	    1.00000	    0.00000	 138.0	 1	    1.06000	    0.94000;
	34	 2	 59.0	 26.0	 0.0	 14.0	 1	    1.00000	    0.00000	 138.0	 1	    1.06000	    0.94000;
	35	 1	 33.0	 9.0	 0.0	 0.0	 1	    1.00000	    0.00000	 138.0	 1	    1.06000	    0.94000;
	36	 2	 31.0	 17.0	 0.0	 0.0	 1	    1.00000	    0.00000	 138.0	 1	    1.06000	    0.94000;
	37	 1	 0.0	 0.0	 0.0	 -25.0	 1	    1.00000	    0.00000	 138.0	 1	    1.06000	    0.94000;
	38	 1	 0.0	 0.0	 0.0	 0.0	 1	    1.00000	    0.00000	 345.0	 1	    1.06000	    0.94000;
	39	 1	 27.0	 11.0	 0.0	 0.0	 1	    1.00000	    0.00000	 138.0	 1	    1.06000	    0.94000;
	40	 2	 66.0	 23.0	 0.0	 0.0	 1	    1.00000	    0.00000	 138.0	 1	    1.06000	    0.94000;
	41	 1	 37.0	 10.0	 0.0	 0.0	 1	    1.00000	    0.00000	 138.0	 1	    1.06000	    0.94000;
	42	 2	 96.0	 23.0	 0.0	 0.0	 1	    1.00000	    0.00000	 138.0	 1	    1.06000	    0.94000;
	43	 1	 18.0	 7.0	 0.0	 0.0	 1	    1.00000	    0.00000	 138.0	 1	    1.06000	    0.94000;
	44	 1	 16.0	 8.0	 0.0	 10.0	 1	    1.00000	    0.00000	 138.0	 1	    1.06000	    0.94000;
	45	 1	 53.0	 22.0	 0.0	 10.0	 1	    1.00000	    0.00000	 138.0	 1	    1.06000	    0.94000;
	46	 2	 28.0	 10.0	 0.0	 10.0	 1	    1.00000	    0.00000	 138.0	 1	    1.06000	    0.94000;
	47	 1	 34.0	 0.0	 0.0	 0.0	 1	    1.00000	    0.00000	 138.0	 1	    1.06000	    0.94000;
	48	 1	 20.0	 11.0	 0.0	 15.0	 1	    1.00000	    0.00000	 138.0	 1	    1.06000	    0.94000;
	49	 2	 87.0	 30.0	 0.0	 0.0	 1	    1.00000	    0.00000	 138.0	 1	    1.06000	    0.94000;
	50	 1	 17.0	 4.0	 0.0	 0.0	 1	    1.00000	    0.00000	 138.0	 1	    1.06000	    0.94000;
	51	 1	 17.0	 8.0	 0.0	 0.0	 1	    1.00000	    0.00000	 138.0	 1	    1.06000	    0.94000;
	52	 1	 18.0	 5.0	 0.0	 0.0	 1	    1.00000	    0.00000	 138.0	 1	    1.06000	    0.94000;
	53	 1	 23.0	 11.0	 0.0	 0.0	 1	    1.00000	    0.00000	 138.0	 1	    1.06000	    0.94000;
	54	 2	 113.0	 32.0	 0.0	 0.0	 1	    1.00000	    0.00000	 138.0	 1	    1.06000	    0.94000;
	55	 2	 63.0	 22.0	 0.0	 0.0	 1	    1.00000	    0.00000	 138.0	 1	    1.06000	    0.94000;
	56	 2	 84.0	 18.0	 0.0	 0.0	 1	    1.00000	    0.00000	 138.0	 1	    1.06000	    0.94000;
	57	 1	 12.0	 3.0	 0.0	 0.0	 1	    1.00000	    0.00000	 138.0	 1	    1.06000	    0.94000;
	58	 1	 12.0	 3.0	 0.0	 0.0	 1	    1.00000	    0.00000	 138.0	 1	    1.06000	    0.94000;
	59	 2	 277.0	 113.0	 0.0	 0.0	 1	    1.00000	    0.00000	 138.0	 1	    1.06000	    0.94000;
	60	 1	 78.0	 3.0	 0.0	 0.0	 1	    1.00000	    0.00000	 138.0	 1	    1.06000	    0.94000;
	61	 2	 0.0	 0.0	 0.0	 0.0	 1	    1.00000	    0.00000	 138.0	 1	    1.06000	    0.94000;
	62	 2	 77.0	 14.0	 0.0	 0.0	 1	    1.00000	    0.00000	 138.0	 1	    1.06000	    0.94000;
	63	 1	 0.0	 0.0	 0.0	 0.0	 1	    1.00000	    0.00000	 345.0	 1	    1.06000	    0.94000;
	64	 1	 0.0	 0.0	 0.0	 0.0	 1	    1.00000	    0.00000	 345.0	 1	    1.06000	    0.94000;
	65	 2	 0.0	 0.0	 0.0	 0.0	 1	    1.00000	    0.00000	 345.0	 1	    1.06000	    0.94000;
	66	 2	 39.0	 18.0	 0.0	 0.0	 1	    1.00000	    0.00000	 138.0	 1	    1.06000	    0.94000;
	67	 1	 28.0	 7.0	 0.0	 0.0	 1	    1.00000	    0.00000	 138.0	 1	    1.06000	    0.94000;
	68	 1	 0.0	 0.0	 0.0	 0.0	 1	    1.00000	    0.00000	 345.0	 1	    1.06000	    0.94000;
	69	 3	 0.0	 0.0	 0.0	 0.0	 1	    1.00000	    0.00000	 138.0	 1	    1.06000	    0.94000;
	70	 2	 66.0	 20.0	 0.0	 0.0	 1	    1.00000	    0.00000	 138.0	 1	    1.06000	    0.94000;
	71	 1	 0.0	 0.0	 0.0	 0.0	 1	    1.00000	    0.00000	 138.0	 1	    1.06000	    0.94000;
	72	 2	 12.0	 0.0	 0.0	 0.0	 1	    1.00000	    0.00000	 138.0	 1	    1.06000	    0.94000;
	73	 2	 6.0	 0.0	 0.0	 0.0	 1	    1.00000	    0.00000	 138.0	 1	    1.06000	    0.94000;
	74	 2	 68.0	 27.0	 0.0	 12.0	 1	    1.00000	    0.00000	 138.0	 1	    1.06000	    0.94000;
	75	 1	 47.0	 11.0	 0.0	 0.0	 1	    1.00000	    0.00000	 138.0	 1	    1.06000	    0.94000;
	76	 2	 68.0	 36.0	 0.0	 0.0	 1	    1.00000	    0.00000	 138.0	 1	    1.06000	    0.94000;
	77	 2	 61.0	 28.0	 0.0	 0.0	 1	    1.00000	    0.00000	 138.0	 1	    1.06000	    0.94000;
	78	 1	 71.0	 26.0	 0.0	 0.0	 1	    1.00000	    0.00000	 138.0	 1	    1.06000	    0.94000;
	79	 1	 39.0	 32.0	 0.0	 20.0	 1	    1.00000	    0.00000	 138.0	 1	    1.06000	    0.94000;
	80	 2	 130.0	 26.0	 0.0	 0.0	 1	    1.00000	    0.00000	 138.0	 1	    1.06000	    0.94000;
	81	 1	 0.0	 0.0	 0.0	 0.0	 1	    1.00000	    0.00000	 345.0	 1	    1.06000	    0.94000;
	82	 1	 54.0	 27.0	 0.0	 20.0	 1	    1.00000	    0.00000	 138.0	 1	    1.06000	    0.94000;
	83	 1	 20.0	 10.0	 0.0	 10.0	 1	    1.00000	    0.00000	 138.0	 1	    1.06000	    0.94000;
	84	 1	 11.0	 7.0	 0.0	 0.0	 1	    1.00000	    0.00000	 138.0	 1	    1.06000	    0.94000;
	85	 2	 24.0	 15.0	 0.0	 0.0	 1	    1.00000	    0.00000	 138.0	 1	    1.06000	    0.94000;
	86	 1	 21.0	 10.0	 0.0	 0.0	 1	    1.00000	    0.00000	 138.0	 1	    1.06000	    0.94000;
	87	 2	 0.0	 0.0	 0.0	 0.0	 1	    1.00000	    0.00000	 161.0	 1	    1.06000	    0.94000;
	88	 1	 48.0	 10.0	 0.0	 0.0	 1	    1.00000	    0.00000	 138.0	 1	    1.06000	    0.94000;
	89	 2	 0.0	 0.0	 0.0	 0.0	 1	    1.00000	    0.00000	 138.0	 1	    1.06000	    0.94000;
	90	 2	 163.0	 42.0	 0.0	 0.0	 1	    1.00000	    0.00000	 138.0	 1	    1.06000	    0.94000;
	91	 2	 10.0	 0.0	 0.0	 0.0	 1	    1.00000	    0.00000	 138.0	 1	    1.06000	    0.94000;
	92	 2	 65.0	 10.0	 0.0	 0.0	 1	    1.00000	    0.00000	 138.0	 1	    1.06000	    0.94000;
	93	 1	 12.0	 7.0	 0.0	 0.0	 1	    1.00000	    0.00000	 138.0	 1	    1.06000	    0.94000;
	94	 1	 30.0	 16.0	 0.0	 0.0	 1	    1.00000	    0.00000	 138.0	 1	    1.06000	    0.94000;
	95	 1	 42.0	 31.0	 0.0	 0.0	 1	    1.00000	    0.00000	 138.0	 1	    1.06000	    0.94000;
	96	 1	 38.0	 15.0	 0.0	 0.0	 1	    1.00000	    0.00000	 138.0	 1	    1.06000	    0.94000;
	97	 1	 15.0	 9.0	 0.0	 0.0	 1	    1.00000	    0.00000	 138.0	 1	    1.06000	    0.94000;
	98	 1	 34.0	 8.0	 0.0	 0.0	 1	    1.00000	    0.00000	 138.0	 1	    1.06000	    0.94000;
	99	 2	 42.0	 0.0	 0.0	 0.0	 1	    1.00000	    0.00000	 138.0	 1	    1.06000	    0.94000;
	100	 2	 37.0	 18.0	 0.0	 0.0	 1	    1.00000	    0.00000	 138.0	 1	    1.06000	    0.94000;
	101	 1	 22.0	 15.0	 0.0	 0.0	 1	    1.00000	    0.00000	 138.0	 1	    1.06000	    0.94000;
	102	 1	 5.0	 3.0	 0.0	 0.0	 1	    1.00000	    0.00000	 138.0	 1	    1.06000	    0.94000;
	103	 2	 23.0	 16.0	 0.0	 0.0	 1	    1.00000	    0.00000	 138.0	 1	    1.06000	    0.94000;
	104	 2	 38.0	 25.0	 0.0	 0.0	 1	    1.00000	    0.00000	 138.0	 1	    1.06000	    0.94000;
	105	 2	 31.0	 26.0	 0.0	 20.0	 1	    1.00000	    0.00000	 138.0	 1	    1.06000	    0.94000;
	106	 1	 43.0	 16.0	 0.0	 0.0	 1	    1.00000	    0.00000	 138.0	 1	    1.06000	    0.94000;
	107	 2	 50.0	 12.0	 0.0	 6.0	 1	    1.00000	    0.00000	 138.0	 1	    1.06000	    0.94000;
	108	 1	 2.0	 1.0	 0.0	 0.0	 1	    1.00000	    0.00000	 138.0	 1	    1.06000	    0.94000;
	109	 1	 8.0	 3.0	 0.0	 0.0	 1	    1.00000	    0.00000	 138.0	 1	    1.06000	    0.94000;
	110	 2	 39.0	 30.0	 0.0	 6.0	 1	    1.00000	    0.00000	 138.0	 1	    1.06000	    0.94000;
	111	 2	 0.0	 0.0	 0.0	 0.0	 1	    1.00000	    0.00000	 138.0	 1	    1.06000	    0.94000;
	112	 2	 68.0	 13.0	 0.0	 0.0	 1	    1.00000	    0.00000	 138.0	 1	    1.06000	    0.94000;
	113	 2	 6.0	 0.0	 0.0	 0.0	 1	    1.00000	    0.00000	 138.0	 1	    1.06000	    0.94000;
	114	 1	 8.0	 3.0	 0.0	 0.0	 1	    1.00000	    0.00000	 138.0	 1	    1.06000	    0.94000;
	115	 1	 22.0	 7.0	 0.0	 0.0	 1	    1.00000	    0.00000	 138.0	 1	    1.06000	    0.94000;
	116	 2	 184.0	 0.0	 0.0	 0.0	 1	    1.00000	    0.00000	 138.0	 1	    1.06000	    0.94000;
	117	 1	 20.0	 8.0	 0.0	 0.0	 1	    1.00000	    0.00000	 138.0	 1	    1.06000	    0.94000;
	118	 1	 33.0	 15.0	 0.0	 0.0	 1	    1.00000	    0.00000	 138.0	 1	    1.06000	    0.94000;
];

%% generator data
%	bus	Pg	Qg	Qmax	Qmin	Vg	mBase	status	Pmax	Pmin
mpc.gen = [
	1	 0.0	 5.0	 15.0	 -5.0	 1.0	 100.0	 1	 0.0	 0.0;
	4	 0.0	 0.0	 300.0	 -300.0	 1.0	 100.0	 1	 0.0	 0.0;
	6	 0.0	 18.5	 50.0	 -13.0	 1.0	 100.0	 1	 0.0	 0.0;
	8	 0.0	 0.0	 300.0	 -300.0	 1.0	 100.0	 1	 0.0	 0.0;
	10	 252.5	 26.5	 200.0	 -147.0	 1.0	 100.0	 1	 505.0	 0.0;
	12	 42.5	 4.0	 43.0	 -35.0	 1.0	 100.0	 1	 85.0	 0.0;
	15	 0.0	 10.0	 30.0	 -10.0	 1.0	 100.0	 1	 0.0	 0.0;
	18	 0.0	 17.0	 50.0	 -16.0	 1.0	 100.0	 1	 0.0	 0.0;
	19	 0.0	 8.0	 24.0	 -8.0	 1.0	 100.0	 1	 0.0	 0.0;
	24	 0.0	 0.0	 300.0	 -300.0	 1.0	 100.0	 1	 0.0	 0.0;
	25	 110.5	 32.0	 111.0	 -47.0	 1.0	 100.0	 1	 221.0	 0.0;
	26	 242.5	 0.0	 243.0	 -243.0	 1.0	 100.0	 1	 485.0	 0.0;
	27	 0.0	 0.0	 300.0	 -300.0	 1.0	 100.0	 1	 0.0	 0.0;
	31	 8.5	 0.0	 9.0	 -9.0	 1.0	 100.0	 1	 17.0	 0.0;
	32	 0.0	 14.0	 42.0	 -14.0	 1.0	 100.0	 1	 0.0	 0.0;
	34	 0.0	 8.0	 24.0	 -8.0	 1.0	 100.0	 1	 0.0	 0.0;
	36	 0.0	 8.0	 24.0	 -8.0	 1.0	 100.0	 1	 0.0	 0.0;
	40	 0.0	 0.0	 300.0	 -300.0	 1.0	 100.0	 1	 0.0	 0.0;
	42	 0.0	 0.0	 300.0	 -300.0	 1.0	 100.0	 1	 0.0	 0.0;
	46	 10.0	 0.0	 10.0	 -10.0	 1.0	 100.0	 1	 20.0	 0.0;
	49	 111.5	 13.5	 112.0	 -85.0	 1.0	 100.0	 1	 223.0	 0.0;
	54	 26.5	 0.0	 27.0	 -27.0	 1.0	 100.0	 1	 53.0	 0.0;
	55	 0.0	 7.5	 23.0	 -8.0	 1.0	 100.0	 1	 0.0	 0.0;
	56	 0.0	 3.5	 15.0	 -8.0	 1.0	 100.0	 1	 0.0	 0.0;
	59	 154.0	 47.0	 154.0	 -60.0	 1.0	 100.0	 1	 308.0	 0.0;
	61	 97.5	 0.0	 98.0	 -98.0	 1.0	 100.0	 1	 195.0	 0.0;
	62	 0.0	 0.0	 20.0	 -20.0	 1.0	 100.0	 1	 0.0	 0.0;
	65	 220.5	 66.5	 200.0	 -67.0	 1.0	 100.0	 1	 441.0	 0.0;
	66	 392.0	 66.5	 200.0	 -67.0	 1.0	 100.0	 1	 784.0	 0.0;
	69	 591.0	 0.0	 300.0	 -300.0	 1.0	 100.0	 1	 1182.0	 0.0;
	70	 0.0	 11.0	 32.0	 -10.0	 1.0	 100.0	 1	 0.0	 0.0;
	72	 0.0	 0.0	 100.0	 -100.0	 1.0	 100.0	 1	 0.0	 0.0;
	73	 0.0	 0.0	 100.0	 -100.0	 1.0	 100.0	 1	 0.0	 0.0;
	74	 0.0	 1.5	 9.0	 -6.0	 1.0	 100.0	 1	 0.0	 0.0;
	76	 0.0	 7.5	 23.0	 -8.0	 1.0	 100.0	 1	 0.0	 0.0;
	77	 0.0	 25.0	 70.0	 -20.0	 1.0	 100.0	 1	 0.0	 0.0;
	80	 254.5	 45.0	 255.0	 -165.0	 1.0	 100.0	 1	 509.0	 0.0;
	85	 0.0	 7.5	 23.0	 -8.0	 1.0	 100.0	 1	 0.0	 0.0;
	87	 5.0	 0.0	 5.0	 -5.0	 1.0	 100.0	 1	 10.0	 0.0;
	89	 318.5	 45.0	 300.0	 -210.0	 1.0	 100.0	 1	 637.0	 0.0;
	90	 0.0	 0.0	 300.0	 -300.0	 1.0	 100.0	 1	 0.0	 0.0;
	91	 0.0	 0.0	 100.0	 -100.0	 1.0	 100.0	 1	 0.0	 0.0;
	92	 0.0	 3.0	 9.0	 -3.0	 1.0	 100.0	 1	 0.0	 0.0;
	99	 0.0	 0.0	 100.0	 -100.0	 1.0	 100.0	 1	 0.0	 0.0;
	100	 326.5	 52.5	 155.0	 -50.0	 1.0	 100.0	 1	 653.0	 0.0;
	103	 54.0	 12.5	 40.0	 -15.0	 1.0	 100.0	 1	 108.0	 0.0;
	104	 0.0	 7.5	 23.0	 -8.0	 1.0	 100.0	 1	 0.0	 0.0;
	105	 0.0	 7.5	 23.0	 -8.0	 1.0	 100.0	 1	 0.0	 0.0;
	107	 0.0	 0.0	 200.0	 -200.0	 1.0	 100.0	 1	 0.0	 0.0;
	110	 0.0	 7.5	 23.0	 -8.0	 1.0	 100.0	 1	 0.0	 0.0;
	111	 39.5	 0.0	 40.0	 -40.0	 1.0	 100.0	 1	 79.0	 0.0;
	112	 0.0	 450.0	 1000.0	 -100.0	 1.0	 100.0	 1	 0.0	 0.0;
	113	 0.0	 50.0	 200.0	 -100.0	 1.0	 100.0	 1	 0.0	 0.0;
	116	 0.0	 0.0	 1000.0	 -1000.0	 1.0	 100.0	 1	 0.0	 0.0;
];

%% generator cost data
%	2	startup	shutdown	n	c(n-1)	...	c0
mpc.gencost = [
	2	 0.0	 0.0	 3	   0.000000	   0.000000	   0.000000;
	2	 0.0	 0.0	 3	   0.000000	   0.000000	   0.000000;
	2	 0.0	 0.0	 3	   0.000000	   0.000000	   0.000000;
	2	 0.0	 0.0	 3	   0.000000	   0.000000	   0.000000;
	2	 0.0	 0.0	 3	   0.000000	  24.983420	   0.000000;
	2	 0.0	 0.0	 3	   0.000000	 124.581564	   0.000000;
	2	 0.0	 0.0	 3	   0.000000	   0.000000	   0.000000;
	2	 0.0	 0.0	 3	   0.000000	   0.000000	   0.000000;
	2	 0.0	 0.0	 3	   0.000000	   0.000000	   0.000000;
	2	 0.0	 0.0	 3	   0.000000	   0.000000	   0.000000;
	2	 0.0	 0.0	 3	   0.000000	  28.948321	   0.000000;
	2	 0.0	 0.0	 3	   0.000000	  22.220980	   0.000000;
	2	 0.0	 0.0	 3	   0.000000	   0.000000	   0.000000;
	2	 0.0	 0.0	 3	   0.000000	  25.993982	   0.000000;
	2	 0.0	 0.0	 3	   0.000000	   0.000000	   0.000000;
	2	 0.0	 0.0	 3	   0.000000	   0.000000	   0.000000;
	2	 0.0	 0.0	 3	   0.000000	   0.000000	   0.000000;
	2	 0.0	 0.0	 3	   0.000000	   0.000000	   0.000000;
	2	 0.0	 0.0	 3	   0.000000	   0.000000	   0.000000;
	2	 0.0	 0.0	 3	   0.000000	  24.202306	   0.000000;
	2	 0.0	 0.0	 3	   0.000000	  16.673942	   0.000000;
	2	 0.0	 0.0	 3	   0.000000	  27.277343	   0.000000;
	2	 0.0	 0.0	 3	   0.000000	   0.000000	   0.000000;
	2	 0.0	 0.0	 3	   0.000000	   0.000000	   0.000000;
	2	 0.0	 0.0	 3	   0.000000	  24.861868	   0.000000;
	2	 0.0	 0.0	 3	   0.000000	  16.056042	   0.000000;
	2	 0.0	 0.0	 3	   0.000000	   0.000000	   0.000000;
	2	 0.0	 0.0	 3	   0.000000	  34.781778	   0.000000;
	2	 0.0	 0.0	 3	   0.000000	  32.668781	   0.000000;
	2	 0.0	 0.0	 3	   0.000000	  25.758442	   0.000000;
	2	 0.0	 0.0	 3	   0.000000	   0.000000	   0.000000;
	2	 0.0	 0.0	 3	   0.000000	   0.000000	   0.000000;
	2	 0.0	 0.0	 3	   0.000000	   0.000000	   0.000000;
	2	 0.0	 0.0	 3	   0.000000	   0.000000	   0.000000;
	2	 0.0	 0.0	 3	   0.000000	   0.000000	   0.000000;
	2	 0.0	 0.0	 3	   0.000000	   0.000000	   0.000000;
	2	 0.0	 0.0	 3	   0.000000	  24.600772	   0.000000;
	2	 0.0	 0.0	 3	   0.000000	   0.000000	   0.000000;
	2	 0.0	 0.0	 3	   0.000000	  34.072633	   0.000000;
	2	 0.0	 0.0	 3	   0.000000	  24.605102	   0.000000;
	2	 0.0	 0.0	 3	   0.000000	   0.000000	   0.000000;
	2	 0.0	 0.0	 3	   0.000000	   0.000000	   0.000000;
	2	 0.0	 0.0	 3	   0.000000	   0.000000	   0.000000;
	2	 0.0	 0.0	 3	   0.000000	   0.000000	   0.000000;
	2	 0.0	 0.0	 3	   0.000000	  12.612170	   0.000000;
	2	 0.0	 0.0	 3	   0.000000	  28.649471	   0.000000;
	2	 0.0	 0.0	 3	   0.000000	   0.000000	   0.000000;
	2	 0.0	 0.0	 3	   0.000000	   0.000000	   0.000000;
	2	 0.0	 0.0	 3	   0.000000	   0.000000	   0.000000;
	2	 0.0	 0.0	 3	   0.000000	   0.000000	   0.000000;
	2	 0.0	 0.0	 3	   0.000000	  35.043401	   0.000000;
	2	 0.0	 0.0	 3	   0.000000	   0.000000	   0.000000;
	2	 0.0	 0.0	 3	   0.000000	   0.000000	   0.000000;
	2	 0.0	 0.0	 3	   0.000000	   0.000000	   0.000000;
];

%% branch data
%	fbus	tbus	r	x	b	rateA	rateB	rateC	ratio	angle	status	angmin	angmax
mpc.branch = [
	1	 2	 0.0303	 0.0999	 0.0254	 151.0	 151.0	 151.0	 0.0	 0.0	 1	 -10.4187716451	 10.4187716451;
	1	 3	 0.0129	 0.0424	 0.01082	 151.0	 151.0	 151.0	 0.0	 0.0	 1	 -10.4187716451	 10.4187716451;
	4	 5	 0.00176	 0.00798	 0.0021	 176.0	 176.0	 176.0	 0.0	 0.0	 1	 -10.4187716451	 10.4187716451;
	3	 5	 0.0241	 0.108	 0.0284	 175.0	 175.0	 175.0	 0.0	 0.0	 1	 -10.4187716451	 10.4187716451;
	5	 6	 0.0119	 0.054	 0.01426	 176.0	 176.0	 176.0	 0.0	 0.0	 1	 -10.4187716451	 10.4187716451;
	6	 7	 0.00459	 0.0208	 0.0055	 176.0	 176.0	 176.0	 0.0	 0.0	 1	 -10.4187716451	 10.4187716451;
	8	 9	 0.00244	 0.0305	 1.162	 711.0	 711.0	 711.0	 0.0	 0.0	 1	 -10.4187716451	 10.4187716451;
	8	 5	 0.0	 0.0267	 0.0	 1099.0	 1099.0	 1099.0	 0.985	 0.0	 1	 -10.4187716451	 10.4187716451;
	9	 10	 0.00258	 0.0322	 1.23	 710.0	 710.0	 710.0	 0.0	 0.0	 1	 -10.4187716451	 10.4187716451;
	4	 11	 0.0209	 0.0688	 0.01748	 151.0	 151.0	 151.0	 0.0	 0.0	 1	 -10.4187716451	 10.4187716451;
	5	 11	 0.0203	 0.0682	 0.01738	 152.0	 152.0	 152.0	 0.0	 0.0	 1	 -10.4187716451	 10.4187716451;
	11	 12	 0.00595	 0.0196	 0.00502	 151.0	 151.0	 151.0	 0.0	 0.0	 1	 -10.4187716451	 10.4187716451;
	2	 12	 0.0187	 0.0616	 0.01572	 151.0	 151.0	 151.0	 0.0	 0.0	 1	 -10.4187716451	 10.4187716451;
	3	 12	 0.0484	 0.16	 0.0406	 151.0	 151.0	 151.0	 0.0	 0.0	 1	 -10.4187716451	 10.4187716451;
	7	 12	 0.00862	 0.034	 0.00874	 164.0	 164.0	 164.0	 0.0	 0.0	 1	 -10.4187716451	 10.4187716451;
	11	 13	 0.02225	 0.0731	 0.01876	 151.0	 151.0	 151.0	 0.0	 0.0	 1	 -10.4187716451	 10.4187716451;
	12	 14	 0.0215	 0.0707	 0.01816	 151.0	 151.0	 151.0	 0.0	 0.0	 1	 -10.4187716451	 10.4187716451;
	13	 15	 0.0744	 0.2444	 0.06268	 115.0	 115.0	 115.0	 0.0	 0.0	 1	 -10.4187716451	 10.4187716451;
	14	 15	 0.0595	 0.195	 0.0502	 144.0	 144.0	 144.0	 0.0	 0.0	 1	 -10.4187716451	 10.4187716451;
	12	 16	 0.0212	 0.0834	 0.0214	 164.0	 164.0	 164.0	 0.0	 0.0	 1	 -10.4187716451	 10.4187716451;
	15	 17	 0.0132	 0.0437	 0.0444	 151.0	 151.0	 151.0	 0.0	 0.0	 1	 -10.4187716451	 10.4187716451;
	16	 17	 0.0454	 0.1801	 0.0466	 158.0	 158.0	 158.0	 0.0	 0.0	 1	 -10.4187716451	 10.4187716451;
	17	 18	 0.0123	 0.0505	 0.01298	 167.0	 167.0	 167.0	 0.0	 0.0	 1	 -10.4187716451	 10.4187716451;
	18	 19	 0.01119	 0.0493	 0.01142	 173.0	 173.0	 173.0	 0.0	 0.0	 1	 -10.4187716451	 10.4187716451;
	19	 20	 0.0252	 0.117	 0.0298	 178.0	 178.0	 178.0	 0.0	 0.0	 1	 -10.4187716451	 10.4187716451;
	15	 19	 0.012	 0.0394	 0.0101	 151.0	 151.0	 151.0	 0.0	 0.0	 1	 -10.4187716451	 10.4187716451;
	20	 21	 0.0183	 0.0849	 0.0216	 177.0	 177.0	 177.0	 0.0	 0.0	 1	 -10.4187716451	 10.4187716451;
	21	 22	 0.0209	 0.097	 0.0246	 178.0	 178.0	 178.0	 0.0	 0.0	 1	 -10.4187716451	 10.4187716451;
	22	 23	 0.0342	 0.159	 0.0404	 178.0	 178.0	 178.0	 0.0	 0.0	 1	 -10.4187716451	 10.4187716451;
	23	 24	 0.0135	 0.0492	 0.0498	 158.0	 158.0	 158.0	 0.0	 0.0	 1	 -10.4187716451	 10.4187716451;
	23	 25	 0.0156	 0.08	 0.0864	 186.0	 186.0	 186.0	 0.0	 0.0	 1	 -10.4187716451	 10.4187716451;
	26	 25	 0.0	 0.0382	 0.0	 768.0	 768.0	 768.0	 0.96	 0.0	 1	 -10.4187716451	 10.4187716451;
	25	 27	 0.0318	 0.163	 0.1764	 177.0	 177.0	 177.0	 0.0	 0.0	 1	 -10.4187716451	 10.4187716451;
	27	 28	 0.01913	 0.0855	 0.0216	 174.0	 174.0	 174.0	 0.0	 0.0	 1	 -10.4187716451	 10.4187716451;
	28	 29	 0.0237	 0.0943	 0.0238	 165.0	 165.0	 165.0	 0.0	 0.0	 1	 -10.4187716451	 10.4187716451;
	30	 17	 0.0	 0.0388	 0.0	 756.0	 756.0	 756.0	 0.96	 0.0	 1	 -10.4187716451	 10.4187716451;
	8	 30	 0.00431	 0.0504	 0.514	 580.0	 580.0	 580.0	 0.0	 0.0	 1	 -10.4187716451	 10.4187716451;
	26	 30	 0.00799	 0.086	 0.908	 340.0	 340.0	 340.0	 0.0	 0.0	 1	 -10.4187716451	 10.4187716451;
	17	 31	 0.0474	 0.1563	 0.0399	 151.0	 151.0	 151.0	 0.0	 0.0	 1	 -10.4187716451	 10.4187716451;
	29	 31	 0.0108	 0.0331	 0.0083	 146.0	 146.0	 146.0	 0.0	 0.0	 1	 -10.4187716451	 10.4187716451;
	23	 32	 0.0317	 0.1153	 0.1173	 158.0	 158.0	 158.0	 0.0	 0.0	 1	 -10.4187716451	 10.4187716451;
	31	 32	 0.0298	 0.0985	 0.0251	 151.0	 151.0	 151.0	 0.0	 0.0	 1	 -10.4187716451	 10.4187716451;
	27	 32	 0.0229	 0.0755	 0.01926	 151.0	 151.0	 151.0	 0.0	 0.0	 1	 -10.4187716451	 10.4187716451;
	15	 33	 0.038	 0.1244	 0.03194	 150.0	 150.0	 150.0	 0.0	 0.0	 1	 -10.4187716451	 10.4187716451;
	19	 34	 0.0752	 0.247	 0.0632	 114.0	 114.0	 114.0	 0.0	 0.0	 1	 -10.4187716451	 10.4187716451;
	35	 36	 0.00224	 0.0102	 0.00268	 176.0	 176.0	 176.0	 0.0	 0.0	 1	 -10.4187716451	 10.4187716451;
	35	 37	 0.011	 0.0497	 0.01318	 175.0	 175.0	 175.0	 0.0	 0.0	 1	 -10.4187716451	 10.4187716451;
	33	 37	 0.0415	 0.142	 0.0366	 154.0	 154.0	 154.0	 0.0	 0.0	 1	 -10.4187716451	 10.4187716451;
	34	 36	 0.00871	 0.0268	 0.00568	 146.0	 146.0	 146.0	 0.0	 0.0	 1	 -10.4187716451	 10.4187716451;
	34	 37	 0.00256	 0.0094	 0.00984	 159.0	 159.0	 159.0	 0.0	 0.0	 1	 -10.4187716451	 10.4187716451;
	38	 37	 0.0	 0.0375	 0.0	 783.0	 783.0	 783.0	 0.935	 0.0	 1	 -10.4187716451	 10.4187716451;
	37	 39	 0.0321	 0.106	 0.027	 151.0	 151.0	 151.0	 0.0	 0.0	 1	 -10.4187716451	 10.4187716451;
	37	 40	 0.0593	 0.168	 0.042	 140.0	 140.0	 140.0	 0.0	 0.0	 1	 -10.4187716451	 10.4187716451;
	30	 38	 0.00464	 0.054	 0.422	 542.0	 542.0	 542.0	 0.0	 0.0	 1	 -10.4187716451	 10.4187716451;
	39	 40	 0.0184	 0.0605	 0.01552	 151.0	 151.0	 151.0	 0.0	 0.0	 1	 -10.4187716451	 10.4187716451;
	40	 41	 0.0145	 0.0487	 0.01222	 152.0	 152.0	 152.0	 0.0	 0.0	 1	 -10.4187716451	 10.4187716451;
	40	 42	 0.0555	 0.183	 0.0466	 151.0	 151.0	 151.0	 0.0	 0.0	 1	 -10.4187716451	 10.4187716451;
	41	 42	 0.041	 0.135	 0.0344	 151.0	 151.0	 151.0	 0.0	 0.0	 1	 -10.4187716451	 10.4187716451;
	43	 44	 0.0608	 0.2454	 0.06068	 117.0	 117.0	 117.0	 0.0	 0.0	 1	 -10.4187716451	 10.4187716451;
	34	 43	 0.0413	 0.1681	 0.04226	 167.0	 167.0	 167.0	 0.0	 0.0	 1	 -10.4187716451	 10.4187716451;
	44	 45	 0.0224	 0.0901	 0.0224	 166.0	 166.0	 166.0	 0.0	 0.0	 1	 -10.4187716451	 10.4187716451;
	45	 46	 0.04	 0.1356	 0.0332	 153.0	 153.0	 153.0	 0.0	 0.0	 1	 -10.4187716451	 10.4187716451;
	46	 47	 0.038	 0.127	 0.0316	 152.0	 152.0	 152.0	 0.0	 0.0	 1	 -10.4187716451	 10.4187716451;
	46	 48	 0.0601	 0.189	 0.0472	 148.0	 148.0	 148.0	 0.0	 0.0	 1	 -10.4187716451	 10.4187716451;
	47	 49	 0.0191	 0.0625	 0.01604	 150.0	 150.0	 150.0	 0.0	 0.0	 1	 -10.4187716451	 10.4187716451;
	42	 49	 0.0715	 0.323	 0.086	 89.0	 89.0	 89.0	 0.0	 0.0	 1	 -10.4187716451	 10.4187716451;
	42	 49	 0.0715	 0.323	 0.086	 89.0	 89.0	 89.0	 0.0	 0.0	 1	 -10.4187716451	 10.4187716451;
	45	 49	 0.0684	 0.186	 0.0444	 138.0	 138.0	 138.0	 0.0	 0.0	 1	 -10.4187716451	 10.4187716451;
	48	 49	 0.0179	 0.0505	 0.01258	 140.0	 140.0	 140.0	 0.0	 0.0	 1	 -10.4187716451	 10.4187716451;
	49	 50	 0.0267	 0.0752	 0.01874	 140.0	 140.0	 140.0	 0.0	 0.0	 1	 -10.4187716451	 10.4187716451;
	49	 51	 0.0486	 0.137	 0.0342	 140.0	 140.0	 140.0	 0.0	 0.0	 1	 -10.4187716451	 10.4187716451;
	51	 52	 0.0203	 0.0588	 0.01396	 142.0	 142.0	 142.0	 0.0	 0.0	 1	 -10.4187716451	 10.4187716451;
	52	 53	 0.0405	 0.1635	 0.04058	 166.0	 166.0	 166.0	 0.0	 0.0	 1	 -10.4187716451	 10.4187716451;
	53	 54	 0.0263	 0.122	 0.031	 177.0	 177.0	 177.0	 0.0	 0.0	 1	 -10.4187716451	 10.4187716451;
	49	 54	 0.073	 0.289	 0.0738	 99.0	 99.0	 99.0	 0.0	 0.0	 1	 -10.4187716451	 10.4187716451;
	49	 54	 0.0869	 0.291	 0.073	 97.0	 97.0	 97.0	 0.0	 0.0	 1	 -10.4187716451	 10.4187716451;
	54	 55	 0.0169	 0.0707	 0.0202	 169.0	 169.0	 169.0	 0.0	 0.0	 1	 -10.4187716451	 10.4187716451;
	54	 56	 0.00275	 0.00955	 0.00732	 155.0	 155.0	 155.0	 0.0	 0.0	 1	 -10.4187716451	 10.4187716451;
	55	 56	 0.00488	 0.0151	 0.00374	 146.0	 146.0	 146.0	 0.0	 0.0	 1	 -10.4187716451	 10.4187716451;
	56	 57	 0.0343	 0.0966	 0.0242	 140.0	 140.0	 140.0	 0.0	 0.0	 1	 -10.4187716451	 10.4187716451;
	50	 57	 0.0474	 0.134	 0.0332	 140.0	 140.0	 140.0	 0.0	 0.0	 1	 -10.4187716451	 10.4187716451;
	56	 58	 0.0343	 0.0966	 0.0242	 140.0	 140.0	 140.0	 0.0	 0.0	 1	 -10.4187716451	 10.4187716451;
	51	 58	 0.0255	 0.0719	 0.01788	 140.0	 140.0	 140.0	 0.0	 0.0	 1	 -10.4187716451	 10.4187716451;
	54	 59	 0.0503	 0.2293	 0.0598	 125.0	 125.0	 125.0	 0.0	 0.0	 1	 -10.4187716451	 10.4187716451;
	56	 59	 0.0825	 0.251	 0.0569	 112.0	 112.0	 112.0	 0.0	 0.0	 1	 -10.4187716451	 10.4187716451;
	56	 59	 0.0803	 0.239	 0.0536	 117.0	 117.0	 117.0	 0.0	 0.0	 1	 -10.4187716451	 10.4187716451;
	55	 59	 0.04739	 0.2158	 0.05646	 133.0	 133.0	 133.0	 0.0	 0.0	 1	 -10.4187716451	 10.4187716451;
	59	 60	 0.0317	 0.145	 0.0376	 176.0	 176.0	 176.0	 0.0	 0.0	 1	 -10.4187716451	 10.4187716451;
	59	 61	 0.0328	 0.15	 0.0388	 176.0	 176.0	 176.0	 0.0	 0.0	 1	 -10.4187716451	 10.4187716451;
	60	 61	 0.00264	 0.0135	 0.01456	 186.0	 186.0	 186.0	 0.0	 0.0	 1	 -10.4187716451	 10.4187716451;
	60	 62	 0.0123	 0.0561	 0.01468	 176.0	 176.0	 176.0	 0.0	 0.0	 1	 -10.4187716451	 10.4187716451;
	61	 62	 0.00824	 0.0376	 0.0098	 176.0	 176.0	 176.0	 0.0	 0.0	 1	 -10.4187716451	 10.4187716451;
	63	 59	 0.0	 0.0386	 0.0	 760.0	 760.0	 760.0	 0.96	 0.0	 1	 -10.4187716451	 10.4187716451;
	63	 64	 0.00172	 0.02	 0.216	 687.0	 687.0	 687.0	 0.0	 0.0	 1	 -10.4187716451	 10.4187716451;
	64	 61	 0.0	 0.0268	 0.0	 1095.0	 1095.0	 1095.0	 0.985	 0.0	 1	 -10.4187716451	 10.4187716451;
	38	 65	 0.00901	 0.0986	 1.046	 297.0	 297.0	 297.0	 0.0	 0.0	 1	 -10.4187716451	 10.4187716451;
	64	 65	 0.00269	 0.0302	 0.38	 675.0	 675.0	 675.0	 0.0	 0.0	 1	 -10.4187716451	 10.4187716451;
	49	 66	 0.018	 0.0919	 0.0248	 186.0	 186.0	 186.0	 0.0	 0.0	 1	 -10.4187716451	 10.4187716451;
	49	 66	 0.018	 0.0919	 0.0248	 186.0	 186.0	 186.0	 0.0	 0.0	 1	 -10.4187716451	 10.4187716451;
	62	 66	 0.0482	 0.218	 0.0578	 132.0	 132.0	 132.0	 0.0	 0.0	 1	 -10.4187716451	 10.4187716451;
	62	 67	 0.0258	 0.117	 0.031	 176.0	 176.0	 176.0	 0.0	 0.0	 1	 -10.4187716451	 10.4187716451;
	65	 66	 0.0	 0.037	 0.0	 793.0	 793.0	 793.0	 0.935	 0.0	 1	 -10.4187716451	 10.4187716451;
	66	 67	 0.0224	 0.1015	 0.02682	 176.0	 176.0	 176.0	 0.0	 0.0	 1	 -10.4187716451	 10.4187716451;
	65	 68	 0.00138	 0.016	 0.638	 686.0	 686.0	 686.0	 0.0	 0.0	 1	 -10.4187716451	 10.4187716451;
	47	 69	 0.0844	 0.2778	 0.07092	 102.0	 102.0	 102.0	 0.0	 0.0	 1	 -10.4187716451	 10.4187716451;
	49	 69	 0.0985	 0.324	 0.0828	 87.0	 87.0	 87.0	 0.0	 0.0	 1	 -10.4187716451	 10.4187716451;
	68	 69	 0.0	 0.037	 0.0	 793.0	 793.0	 793.0	 0.935	 0.0	 1	 -10.4187716451	 10.4187716451;
	69	 70	 0.03	 0.127	 0.122	 170.0	 170.0	 170.0	 0.0	 0.0	 1	 -10.4187716451	 10.4187716451;
	24	 70	 0.00221	 0.4115	 0.10198	 72.0	 72.0	 72.0	 0.0	 0.0	 1	 -10.4187716451	 10.4187716451;
	70	 71	 0.00882	 0.0355	 0.00878	 166.0	 166.0	 166.0	 0.0	 0.0	 1	 -10.4187716451	 10.4187716451;
	24	 72	 0.0488	 0.196	 0.0488	 146.0	 146.0	 146.0	 0.0	 0.0	 1	 -10.4187716451	 10.4187716451;
	71	 72	 0.0446	 0.18	 0.04444	 159.0	 159.0	 159.0	 0.0	 0.0	 1	 -10.4187716451	 10.4187716451;
	71	 73	 0.00866	 0.0454	 0.01178	 188.0	 188.0	 188.0	 0.0	 0.0	 1	 -10.4187716451	 10.4187716451;
	70	 74	 0.0401	 0.1323	 0.03368	 151.0	 151.0	 151.0	 0.0	 0.0	 1	 -10.4187716451	 10.4187716451;
	70	 75	 0.0428	 0.141	 0.036	 151.0	 151.0	 151.0	 0.0	 0.0	 1	 -10.4187716451	 10.4187716451;
	69	 75	 0.0405	 0.122	 0.124	 145.0	 145.0	 145.0	 0.0	 0.0	 1	 -10.4187716451	 10.4187716451;
	74	 75	 0.0123	 0.0406	 0.01034	 151.0	 151.0	 151.0	 0.0	 0.0	 1	 -10.4187716451	 10.4187716451;
	76	 77	 0.0444	 0.148	 0.0368	 152.0	 152.0	 152.0	 0.0	 0.0	 1	 -10.4187716451	 10.4187716451;
	69	 77	 0.0309	 0.101	 0.1038	 150.0	 150.0	 150.0	 0.0	 0.0	 1	 -10.4187716451	 10.4187716451;
	75	 77	 0.0601	 0.1999	 0.04978	 141.0	 141.0	 141.0	 0.0	 0.0	 1	 -10.4187716451	 10.4187716451;
	77	 78	 0.00376	 0.0124	 0.01264	 151.0	 151.0	 151.0	 0.0	 0.0	 1	 -10.4187716451	 10.4187716451;
	78	 79	 0.00546	 0.0244	 0.00648	 174.0	 174.0	 174.0	 0.0	 0.0	 1	 -10.4187716451	 10.4187716451;
	77	 80	 0.017	 0.0485	 0.0472	 141.0	 141.0	 141.0	 0.0	 0.0	 1	 -10.4187716451	 10.4187716451;
	77	 80	 0.0294	 0.105	 0.0228	 157.0	 157.0	 157.0	 0.0	 0.0	 1	 -10.4187716451	 10.4187716451;
	79	 80	 0.0156	 0.0704	 0.0187	 175.0	 175.0	 175.0	 0.0	 0.0	 1	 -10.4187716451	 10.4187716451;
	68	 81	 0.00175	 0.0202	 0.808	 684.0	 684.0	 684.0	 0.0	 0.0	 1	 -10.4187716451	 10.4187716451;
	81	 80	 0.0	 0.037	 0.0	 793.0	 793.0	 793.0	 0.935	 0.0	 1	 -10.4187716451	 10.4187716451;
	77	 82	 0.0298	 0.0853	 0.08174	 141.0	 141.0	 141.0	 0.0	 0.0	 1	 -10.4187716451	 10.4187716451;
	82	 83	 0.0112	 0.03665	 0.03796	 150.0	 150.0	 150.0	 0.0	 0.0	 1	 -10.4187716451	 10.4187716451;
	83	 84	 0.0625	 0.132	 0.0258	 122.0	 122.0	 122.0	 0.0	 0.0	 1	 -10.4187716451	 10.4187716451;
	83	 85	 0.043	 0.148	 0.0348	 154.0	 154.0	 154.0	 0.0	 0.0	 1	 -10.4187716451	 10.4187716451;
	84	 85	 0.0302	 0.0641	 0.01234	 122.0	 122.0	 122.0	 0.0	 0.0	 1	 -10.4187716451	 10.4187716451;
	85	 86	 0.035	 0.123	 0.0276	 156.0	 156.0	 156.0	 0.0	 0.0	 1	 -10.4187716451	 10.4187716451;
	86	 87	 0.02828	 0.2074	 0.0445	 141.0	 141.0	 141.0	 1.0	 0.0	 1	 -10.4187716451	 10.4187716451;
	85	 88	 0.02	 0.102	 0.0276	 186.0	 186.0	 186.0	 0.0	 0.0	 1	 -10.4187716451	 10.4187716451;
	85	 89	 0.0239	 0.173	 0.047	 168.0	 168.0	 168.0	 0.0	 0.0	 1	 -10.4187716451	 10.4187716451;
	88	 89	 0.0139	 0.0712	 0.01934	 186.0	 186.0	 186.0	 0.0	 0.0	 1	 -10.4187716451	 10.4187716451;
	89	 90	 0.0518	 0.188	 0.0528	 151.0	 151.0	 151.0	 0.0	 0.0	 1	 -10.4187716451	 10.4187716451;
	89	 90	 0.0238	 0.0997	 0.106	 169.0	 169.0	 169.0	 0.0	 0.0	 1	 -10.4187716451	 10.4187716451;
	90	 91	 0.0254	 0.0836	 0.0214	 151.0	 151.0	 151.0	 0.0	 0.0	 1	 -10.4187716451	 10.4187716451;
	89	 92	 0.0099	 0.0505	 0.0548	 186.0	 186.0	 186.0	 0.0	 0.0	 1	 -10.4187716451	 10.4187716451;
	89	 92	 0.0393	 0.1581	 0.0414	 166.0	 166.0	 166.0	 0.0	 0.0	 1	 -10.4187716451	 10.4187716451;
	91	 92	 0.0387	 0.1272	 0.03268	 151.0	 151.0	 151.0	 0.0	 0.0	 1	 -10.4187716451	 10.4187716451;
	92	 93	 0.0258	 0.0848	 0.0218	 151.0	 151.0	 151.0	 0.0	 0.0	 1	 -10.4187716451	 10.4187716451;
	92	 94	 0.0481	 0.158	 0.0406	 151.0	 151.0	 151.0	 0.0	 0.0	 1	 -10.4187716451	 10.4187716451;
	93	 94	 0.0223	 0.0732	 0.01876	 151.0	 151.0	 151.0	 0.0	 0.0	 1	 -10.4187716451	 10.4187716451;
	94	 95	 0.0132	 0.0434	 0.0111	 151.0	 151.0	 151.0	 0.0	 0.0	 1	 -10.4187716451	 10.4187716451;
	80	 96	 0.0356	 0.182	 0.0494	 159.0	 159.0	 159.0	 0.0	 0.0	 1	 -10.4187716451	 10.4187716451;
	82	 96	 0.0162	 0.053	 0.0544	 150.0	 150.0	 150.0	 0.0	 0.0	 1	 -10.4187716451	 10.4187716451;
	94	 96	 0.0269	 0.0869	 0.023	 149.0	 149.0	 149.0	 0.0	 0.0	 1	 -10.4187716451	 10.4187716451;
	80	 97	 0.0183	 0.0934	 0.0254	 186.0	 186.0	 186.0	 0.0	 0.0	 1	 -10.4187716451	 10.4187716451;
	80	 98	 0.0238	 0.108	 0.0286	 176.0	 176.0	 176.0	 0.0	 0.0	 1	 -10.4187716451	 10.4187716451;
	80	 99	 0.0454	 0.206	 0.0546	 140.0	 140.0	 140.0	 0.0	 0.0	 1	 -10.4187716451	 10.4187716451;
	92	 100	 0.0648	 0.295	 0.0472	 98.0	 98.0	 98.0	 0.0	 0.0	 1	 -10.4187716451	 10.4187716451;
	94	 100	 0.0178	 0.058	 0.0604	 150.0	 150.0	 150.0	 0.0	 0.0	 1	 -10.4187716451	 10.4187716451;
	95	 96	 0.0171	 0.0547	 0.01474	 149.0	 149.0	 149.0	 0.0	 0.0	 1	 -10.4187716451	 10.4187716451;
	96	 97	 0.0173	 0.0885	 0.024	 186.0	 186.0	 186.0	 0.0	 0.0	 1	 -10.4187716451	 10.4187716451;
	98	 100	 0.0397	 0.179	 0.0476	 160.0	 160.0	 160.0	 0.0	 0.0	 1	 -10.4187716451	 10.4187716451;
	99	 100	 0.018	 0.0813	 0.0216	 175.0	 175.0	 175.0	 0.0	 0.0	 1	 -10.4187716451	 10.4187716451;
	100	 101	 0.0277	 0.1262	 0.0328	 176.0	 176.0	 176.0	 0.0	 0.0	 1	 -10.4187716451	 10.4187716451;
	92	 102	 0.0123	 0.0559	 0.01464	 176.0	 176.0	 176.0	 0.0	 0.0	 1	 -10.4187716451	 10.4187716451;
	101	 102	 0.0246	 0.112	 0.0294	 176.0	 176.0	 176.0	 0.0	 0.0	 1	 -10.4187716451	 10.4187716451;
	100	 103	 0.016	 0.0525	 0.0536	 151.0	 151.0	 151.0	 0.0	 0.0	 1	 -10.4187716451	 10.4187716451;
	100	 104	 0.0451	 0.204	 0.0541	 141.0	 141.0	 141.0	 0.0	 0.0	 1	 -10.4187716451	 10.4187716451;
	103	 104	 0.0466	 0.1584	 0.0407	 153.0	 153.0	 153.0	 0.0	 0.0	 1	 -10.4187716451	 10.4187716451;
	103	 105	 0.0535	 0.1625	 0.0408	 145.0	 145.0	 145.0	 0.0	 0.0	 1	 -10.4187716451	 10.4187716451;
	100	 106	 0.0605	 0.229	 0.062	 124.0	 124.0	 124.0	 0.0	 0.0	 1	 -10.4187716451	 10.4187716451;
	104	 105	 0.00994	 0.0378	 0.00986	 161.0	 161.0	 161.0	 0.0	 0.0	 1	 -10.4187716451	 10.4187716451;
	105	 106	 0.014	 0.0547	 0.01434	 164.0	 164.0	 164.0	 0.0	 0.0	 1	 -10.4187716451	 10.4187716451;
	105	 107	 0.053	 0.183	 0.0472	 154.0	 154.0	 154.0	 0.0	 0.0	 1	 -10.4187716451	 10.4187716451;
	105	 108	 0.0261	 0.0703	 0.01844	 137.0	 137.0	 137.0	 0.0	 0.0	 1	 -10.4187716451	 10.4187716451;
	106	 107	 0.053	 0.183	 0.0472	 154.0	 154.0	 154.0	 0.0	 0.0	 1	 -10.4187716451	 10.4187716451;
	108	 109	 0.0105	 0.0288	 0.0076	 138.0	 138.0	 138.0	 0.0	 0.0	 1	 -10.4187716451	 10.4187716451;
	103	 110	 0.03906	 0.1813	 0.0461	 159.0	 159.0	 159.0	 0.0	 0.0	 1	 -10.4187716451	 10.4187716451;
	109	 110	 0.0278	 0.0762	 0.0202	 138.0	 138.0	 138.0	 0.0	 0.0	 1	 -10.4187716451	 10.4187716451;
	110	 111	 0.022	 0.0755	 0.02	 154.0	 154.0	 154.0	 0.0	 0.0	 1	 -10.4187716451	 10.4187716451;
	110	 112	 0.0247	 0.064	 0.062	 135.0	 135.0	 135.0	 0.0	 0.0	 1	 -10.4187716451	 10.4187716451;
	17	 113	 0.00913	 0.0301	 0.00768	 151.0	 151.0	 151.0	 0.0	 0.0	 1	 -10.4187716451	 10.4187716451;
	32	 113	 0.0615	 0.203	 0.0518	 139.0	 139.0	 139.0	 0.0	 0.0	 1	 -10.4187716451	 10.4187716451;
	32	 114	 0.0135	 0.0612	 0.01628	 176.0	 176.0	 176.0	 0.0	 0.0	 1	 -10.4187716451	 10.4187716451;
	27	 115	 0.0164	 0.0741	 0.01972	 175.0	 175.0	 175.0	 0.0	 0.0	 1	 -10.4187716451	 10.4187716451;
	114	 115	 0.0023	 0.0104	 0.00276	 175.0	 175.0	 175.0	 0.0	 0.0	 1	 -10.4187716451	 10.4187716451;
	68	 116	 0.00034	 0.00405	 0.164	 7218.0	 7218.0	 7218.0	 1.0	 0.0	 1	 -10.4187716451	 10.4187716451;
	12	 117	 0.0329	 0.14	 0.0358	 170.0	 170.0	 170.0	 0.0	 0.0	 1	 -10.4187716451	 10.4187716451;
	75	 118	 0.0145	 0.0481	 0.01198	 151.0	 151.0	 151.0	 0.0	 0.0	 1	 -10.4187716451	 10.4187716451;
	76	 118	 0.0164	 0.0544	 0.01356	 151.0	 151.0	 151.0	 0.0	 0.0	 1	 -10.4187716451	 10.4187716451;
];

% INFO    : === Translation Options ===
% INFO    : Phase Angle Bound:           10.4187716451 (deg.)
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
