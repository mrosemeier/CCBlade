import numpy as np
import matplotlib.pyplot as plt

from ccblade import CCAirfoil, CCBlade
from openmdao.api import Component, Group, Problem
from scipy.optimize import optimize


class CCBladeEvaluator(Component):

    def __init__(self, config, sdim, nalpha, nv, naero_coeffs):
        super(CCBladeEvaluator, self).__init__()

        self.sdim = sdim
        self.nsec = sdim[1]

        self.nalpha = nalpha

        self.nv = nv

        # standard values for cp lambda curves
        self.tsr_cp_start = 1
        self.tsr_cp_end = 18
        self.ntsr_cp = 100
        self.pitches_cp_start = 0
        self.pitches_cp_end = 30
        self.npitches_cp = 10

        #self.blend_var = config['CCBlade']['blend_var']

        self.af_name_base = 'cs_'
        self.af_name_suffix = '_aerodyn'

        self.naero_coeffs = naero_coeffs

        for k, w in config['CCBlade'].iteritems():
            try:
                setattr(self, k, w)
            except:
                pass

        self.nre = len(config['CCBlade']['res'])
        self.nmet = len(config['CCBlade']['analysis_methods'])

        self._init_params()
        self._init_unknowns()

    def _init_params(self):
        self.add_param('blade_length', 0.)

        self.add_param('rotor_diameter', 0.)

        # add planform
        self.add_param('s_st', np.zeros(self.nsec))
        self.add_param('x_st', np.zeros(self.nsec))
        self.add_param('y_st', np.zeros(self.nsec))
        self.add_param('z_st', np.zeros(self.nsec))
        self.add_param('rot_x_st', np.zeros(self.nsec))
        self.add_param('rot_y_st', np.zeros(self.nsec))
        self.add_param('rot_z_st', np.zeros(self.nsec))
        self.add_param('chord_st', np.zeros(self.nsec))
        self.add_param('rthick_st', np.zeros(self.nsec))
        self.add_param('p_le_st', np.zeros(self.nsec))

        self.add_param('cs_polars', np.zeros(  # nsec, alpha, cl ,cd, cm
            (self.nalpha, 4, self.nsec, self.nre, self.nmet)))
        self.add_param(  # amount of polar entries per table
            'n_cs_alpha', np.zeros((self.nsec, self.nre, self.nmet)).astype(int))
        self.add_param('cs_polars_tc', np.zeros((self.nsec)))

        self.add_param('airfoildata:blend_var', np.zeros(self.naero_coeffs))

    def _init_unknowns(self):
        self.add_output('pitch_v', np.zeros((self.nv)))
        self.add_output('aeloads_v', np.zeros((self.nsec, 2, self.nv)))
        self.add_output('omega_v', np.zeros((self.nv)))
        self.add_output('cp_v', np.zeros((self.nv)))
        self.add_output('tsr_v', np.zeros((self.nv)))
        self.add_output('v', np.zeros((self.nv)))
        self.add_output('P_v', np.zeros((self.nv)))

        self.add_output('load_cases_aero', np.zeros((self.nv, self.nsec, 15)))
        self.add_output('load_cases_gravity', np.zeros((self.nv, 1, 15)))
        self.add_output('load_cases_align', np.zeros((self.nv, 5)))

        self.add_output('r_stat', np.zeros((self.nsec)))

        #self.add_output('load_cases_aero_ult', np.zeros((2, self.nsec, 15)))
        #self.add_output('load_cases_align_ult', np.zeros((2, 5)))

        self.add_output('tsr_cp', np.zeros((self.ntsr_cp)))
        self.add_output('CP_cp', np.zeros((self.ntsr_cp, self.npitches_cp)))
        self.add_output('CT_cp', np.zeros((self.ntsr_cp, self.npitches_cp)))
        self.add_output('CQ_cp', np.zeros((self.ntsr_cp, self.npitches_cp)))
        self.add_output('pitches_cp', np.zeros((self.npitches_cp)))

    def solve_nonlinear(self, params, unknowns, resids):

        self.Rtip = 0.5 * params['rotor_diameter']
        self.Rhub = self.Rtip - params['blade_length']
        z = params['z_st'] * params['blade_length']
        self.r = z + self.Rhub

        # First and last section need to be shifted if they coincide with the
        # root/tip
        dr01 = self.r[1] - self.r[0]
        rnode1 = self.r[0] + 0.5 * dr01
        rnode0 = self.r[0]
        self.r[0] = 0.5 * (rnode1 + rnode0)  # override root value

        dr_2_1 = self.r[-1] - self.r[-2]
        rnode_2 = self.r[-2] + 0.5 * dr_2_1
        rnode_1 = self.r[-1]
        self.r[-1] = 0.5 * (rnode_1 + rnode_2)  # override tip value

        unknowns['r_stat'] = self.r

        # Workaround to fix nan calculated by CCblade
        # obsolete
        # self.r[0] = self.Rhub + 1.0E-11
        # self.r[-1] = self.Rtip - 1.0E-11

        rot_z = params['rot_z_st']
        # twist angle
        self.theta = - rot_z

        #file_planform = '/home/becmax/iwes-gitlab/bdtprojects/smartblades/btc_design_model/data/turbine/aero/SmartBlades-Ref12 Aerodynamic Design 1.txt'
        # planform_data = np.loadtxt(
        #    file_planform, skiprows=13, usecols=(0, 1, 3, 4))
        self.chord = params['chord_st'] * params['blade_length']

        self.blend_var = params['airfoildata:blend_var']
        self.af = []
        # init from aerodyn files
        for i, tc in enumerate(self.blend_var):

            af_name = self.af_name_base + \
                '%03d_%04d' % (i, tc * 1000) + self.af_name_suffix

            self.af.append(CCAirfoil.initFromAerodynFile(af_name + '.dat'))

        # alternative input
        '''
        # load all airfoils
        pol = params['cs_polars']  # prob.root.unknowns['af_data']

        naf = len(np.trim_zeros(params['cs_polars_tc']))  # len(af_list)

        #nsec = 0
        #nre = 0

        nmet = 0
        print 'CCBLade uses polars generated with analysis method: %s ' % self.analysis_methods[nmet]

        #cs_polars_met = p['cs_polars_met']
        cs_polars_tc = params['cs_polars_tc']
        #cs_polars_re = p['cs_polars_re']
        n_cs_alpha = params['n_cs_alpha']

        # convert to CCAirfoil
        
        for nsec in range(naf):  # [0, 1, 2, 3, 4, 5, 6, 7]:  #
            re = self.res  # Reynolds numbers where airfoil data are defined
            mins = []
            maxs = []
            for rey in range(self.nre):
                mins.append(np.min(pol[:, 0, nsec, rey, nmet]))
                maxs.append(np.max(pol[:, 0, nsec, rey, nmet]))

            alpha_min = max(mins)
            alpha_max = min(maxs)

            alpha_incr = pol[1, 0, nsec, 0, nmet] - pol[0, 0, nsec, 0, nmet]
            alpha = np.arange(alpha_min, alpha_max + alpha_incr, alpha_incr)

            cl = np.zeros((len(alpha), self.nre))
            cd = np.zeros((len(alpha), self.nre))
            for rey in range(self.nre):
                # cl[i, j] is the lift coefficient at alpha[i] and Re[j]
                cl[:, rey] = np.interp(alpha, pol[:n_cs_alpha[nsec, rey, nmet], 0, nsec, rey, nmet], pol[
                    :n_cs_alpha[nsec, rey, nmet], 1, nsec, rey, nmet])
                # cd[i, j] is the lift coefficient at alpha[i] and Re[j]
                cd[:, rey] = np.interp(alpha, pol[:n_cs_alpha[nsec, rey, nmet], 0, nsec, rey, nmet], pol[
                    :n_cs_alpha[nsec, rey, nmet], 2, nsec, rey, nmet])
            self.af.append(CCAirfoil(alpha, re, cl, cd))
        '''
        # create CCBlade object
        self.rotor = CCBlade(self.r, self.chord, self.theta, self.af, self.Rhub, self.Rtip, self.B, self.rho, self.mu, self.precone, self.tilt, self.yaw, self.shearExp, self.hubHt, self.nSector,
                             presweepTip=0.0, wakerotation=self.wakerotation, usecd=self.usecd, tiploss=self.tiploss, hubloss=self.hubloss, iterRe=self.iterRe)

        # find tsr opt
        cp_opt, tsr_opt, tsr_cp, CP_cp, CT_cp, CQ_cp, pitches_cp = self.evaluate_cp_lambda()
        lambda_opt = tsr_opt[0]  # at pitch zero

        # create wind speed vector
        #U_oper = np.linspace(self.Uin, self.Uout, self.nv)
        U_oper = np.linspace(self.Ustart_eval, self.Ustop_eval, self.nv)

        # create operating conditions
        pitch_angles, cp, aeloads, tsr, omega, P = self.evaluate_steady_states(
            P_elec=self.P_elec, eta_elec=self.eta_elec, lambda_opt=lambda_opt, OmegaMin=self.OmegaMin, OmegaMax=self.OmegaMax, U_oper=U_oper)
        # self.pp.close()

        # get u_rated
        self.diameter = params['rotor_diameter']

        cp_max_idx = cp.argsort()[-1]
        Urated = U_oper[cp_max_idx]
        omega_rated = omega[cp_max_idx]
        pitch_rated = pitch_angles[cp_max_idx]

        # run ultimate load cases
        # DLC 2.3 EOG v_hub = v_r+-2m/s and v_out Ultimate Abnormal
        # acc. to IEC64100-1 ed3 Eq. 17
        #V_hub = Urated
        #V_e1 = 0.8 * V_e50
        #V_g1 = 1.35 * (V_e1 - V_hub)
        #D = self.diameter

        # IEC64100-1 ed Tab.1
        #Iref = self.i_ref

        # IEC64100-1 ed3 Eq. 11
        #b = 5.6
        #sigma1 = Iref * (0.75 * V_hub + b)

        # IEC64100-1 ed3 Eq. 5
        # lambda1 = 0.7*h_hub # z <60m
        # lambda1 = 42  # z <=60m
        #V_g2 = 3.3 * (sigma1 / (1 + 0.1 * (D / lambda1)))
        #V_gust = np.min(np.array([V_g1, V_g2]))

        # DLC 6.1 EWM 50y recurrence period +- 8deg yaw
        shearExp_ult = 0.11

        self.rotor_ult = CCBlade(self.r, self.chord, self.theta, self.af, self.Rhub, self.Rtip, self.B, self.rho, self.mu, self.precone, self.tilt, self.yaw, shearExp_ult, self.hubHt, self.nSector,
                                 presweepTip=0.0, wakerotation=self.wakerotation, usecd=self.usecd, tiploss=self.tiploss, hubloss=self.hubloss, iterRe=self.iterRe)

        V_ref = self.v_ref
        V_e50 = 1.4 * V_ref
        pitch_dlc61 = 90
        yaw_errors = [+8, -8]

        aeloads_ult = self.evaluate_ultimate_loads(
            V_e50, pitch_dlc61, yaw_errors)

        # shift data to unknowns
        unknowns['pitch_v'] = - pitch_angles  # convert positive along z axis
        unknowns['aeloads_v'] = aeloads
        unknowns['omega_v'] = omega
        unknowns['cp_v'] = cp
        unknowns['tsr_v'] = tsr
        unknowns['v'] = U_oper
        unknowns['P_v'] = P

        # convert to FEPROC load case which need to be applied on t/4 point
        # x,y,z,rotx,roty,rotz,Fx,Fy,Fz,Mx,My,Mz,ax,ay,az
        # 'load_cases', np.zeros((self.nv, self.nsec, 15)))

        # tilt, pitch, x_orig, y_orig, z_orig
        #'load_cases_align', np.zeros((self.nv, 5)))

        for i, _ in enumerate(U_oper):
            unknowns['load_cases_aero'][i, :, 2] = params[
                'z_st'] * params['blade_length']
            unknowns['load_cases_aero'][
                i, :, 7] = aeloads[:, 0, i]  # flap-wise
            unknowns['load_cases_aero'][i, :, 6] = aeloads[:, 1, i]  # lead-lag

            unknowns['load_cases_align'][
                i, 1] = - pitch_angles[i]  # positive along z

            # load cases for lead-lag gravity (x-direction)
            unknowns['load_cases_gravity'][i, 0, :] = np.array(
                    [[0, 0, 0,
                      0, 0, 0,
                      9.81, 0.0, 0.0,  # g_x, g_y, gz
                      0.0, 0.0, 0.0,
                      0.0, 0.0, 0.0]])

        # for i, _ in enumerate(yaw_errors):
        #    unknowns['load_cases_aero_ult'][i, :, 2] = params[
         #       'z_st'] * params['blade_length']
         #   unknowns['load_cases_aero_ult'][
         #       i, :, 7] = aeloads_ult[:, 0, i]  # flap-wise
        #    unknowns['load_cases_aero_ult'][
        #        i, :, 6] = aeloads_ult[:, 1, i]  # lead-lag

        #    unknowns['load_cases_align_ult'][
        #        i, 1] = - pitch_dlc61  # positive along z
        unknowns['tsr_cp'] = tsr_cp
        unknowns['CP_cp'] = CP_cp
        unknowns['CT_cp'] = CT_cp
        unknowns['CQ_cp'] = CQ_cp
        unknowns['pitches_cp'] = - pitches_cp  # convert positive along z axis

    def evaluate_cp_lambda(self):
        tsr = np.linspace(self.tsr_cp_start, self.tsr_cp_end, self.ntsr_cp)
        Omega = 10.0 * np.ones_like(tsr)
        Uinf = Omega * np.pi / 30.0 * self.Rtip / tsr

        # pitch = np.zeros_like(tsr)

        pitches = np.linspace(self.pitches_cp_start,
                              self.pitches_cp_end, self.npitches_cp)

        cp_opt = np.zeros_like(pitches)
        tsr_opt = np.zeros_like(pitches)
        CP = np.zeros((len(tsr), len(pitches)))
        CT = np.zeros((len(tsr), len(pitches)))
        CQ = np.zeros((len(tsr), len(pitches)))
        for i, pit in enumerate(pitches):
            pitch = pit * np.ones_like(tsr)
            CP[:, i], CT[:, i], CQ[:, i] = self.rotor.evaluate(
                Uinf, Omega, pitch, coefficient=True)

            cp_opt[i] = np.max(CP[:, i])
            tsr_opt[i] = tsr[CP[:, i].argmax()]

        #filename_excelcoeff = '/home/becmax/iwes-gitlab/bdtprojects/smartblades/btc_design_model/reference_excel_coeff.txt'
        #excel_coeff = np.loadtxt(filename_excelcoeff)

        #TSR_excel = excel_coeff[:, 0]
        #CP_excel = excel_coeff[:, 1]

        return cp_opt, tsr_opt, tsr, CP, CT, CQ, pitches

    def evaluate_partial_load(self, lambda_opt, Urated, OmegaMin, OmegaMax, eta_elec):

        U_vector = np.arange(0, Urated, 0.1)
        Omega_vector = np.zeros_like(U_vector)

        #lambda_opt = 8.4

        for i in range(len(U_vector)):

            Omega = U_vector[i] * 30.0 / np.pi * lambda_opt / self.Rtip
            if Omega < OmegaMin:
                Omega = OmegaMin
            elif Omega > OmegaMax:
                Omega = OmegaMax
            Omega_vector[i] = Omega

        pitch_vector = np.zeros_like(U_vector)
        P, T, Q = self.rotor.evaluate(U_vector, Omega_vector, pitch_vector)

        #filename_excel = '/home/becmax/iwes-gitlab/bdtprojects/smartblades/btc_design_model/reference_excel.txt'
        #excel_data = np.loadtxt(filename_excel)

        #P_excel = excel_data[:, 2]
        #U_excel = excel_data[:, 0]

        plt.figure()
        plt.plot(U_vector, P / eta_elec, label='CCblade')
        #plt.plot(U_excel, P_excel, label='Reference')
        plt.xlabel('Wind speed (m/s)')
        plt.ylabel('Electric power (kW)')
        plt.legend(loc='best')
        plt.grid()
        # self.pp.savefig()

    def evaluate_ultimate_loads(self, V_e50, pitch_dlc61, yaw_errors):

        aeloads_ult = np.zeros((len(self.r), 2, len(yaw_errors)))
        m_tan = []
        m_norm = []
        m_res = []
        for i, yaw_error in enumerate(yaw_errors):
            aeloads_ult[:, 0, i], aeloads_ult[:, 1, i] = self.rotor_ult.distributedAeroLoads(
                V_e50, Omega=0.000, pitch=pitch_dlc61 + yaw_error, azimuth=0)

            #dr = self.r[1] - self.r[0]
            #mn = np.sum(Np * dr * self.r)
            #mt = np.sum(Tp * dr * self.r)
            # gf = 1.35  # normal
            #m_tan.append(-mt * gf / 1E06)
            #m_norm.append(mn * gf / 1E06)
            #mres = np.sqrt(mn**2 + mt**2) * gf / 1E06
            # m_res.append(mres)

        # print m_tan
        # print m_norm
        # print m_res

        return aeloads_ult

    def evaluate_steady_states(self, P_elec, eta_elec, lambda_opt, OmegaMin, OmegaMax, U_oper):

        Uin = self.Uin  # U_oper[0]
        Uout = self.Uout  # U_oper[-1]
        Omega_oper = np.zeros_like(U_oper)
        for i, U in enumerate(U_oper):
            Omega = U_oper[i] * 30.0 / np.pi * lambda_opt / self.Rtip
            if Omega < OmegaMin:
                Omega = OmegaMin
            elif Omega > OmegaMax:
                Omega = OmegaMax

            if U < Uin or U > Uout:
                Omega = 0

            Omega_oper[i] = Omega

        #Omega = 10.0
        #U = 15.0
        P_rated = P_elec / eta_elec
        # start values
        pitch_init = 1.0  # closest

        pitch_angles = np.zeros_like(U_oper)
        cp = np.zeros_like(U_oper)
        P = np.zeros_like(U_oper)
        aeloads = np.zeros((len(self.r), 2, len(U_oper)))
        #Np = np.zeros_like(U_oper)
        #Tp = np.zeros_like(U_oper)
        tsr = np.zeros_like(U_oper)

        for i, (U, Omega) in enumerate(zip(U_oper, Omega_oper)):

            P_goal = P_rated
            if U < Uin or U > Uout:
                P_goal = 0

            # pitch_init2 = pitch_init - 1.0  # second closest
            pitch = pitch_init

            def min_func(pitch):
                ''' function to be minimized
                '''
                """
                U_p_tsr_opt = Omega * np.pi / 30.0 * self.Rtip / lambda_opt
                P_tsr_opt, _, _ = self.rotor.evaluate(
                    [U_p_tsr_opt], [Omega], [pitch])
                if P_tsr_opt < P_rated:
                    P_goal = P_tsr_opt
                """
                P, _, _ = self.rotor.evaluate([U], [Omega], [pitch])
                P_diff = P - P_goal
                return abs(P_diff)

            # do not pitch when omega is controlled
            '''
            if Omega < OmegaMax:
                pitch = 0.
            else:
            '''
            pitch = optimize.brent(
                min_func, brack=(pitch_init, pitch_init + 1))

            cp[i], _, _ = self.rotor.evaluate(
                [U], [Omega], [pitch], coefficient=True)
            P[i], _, _ = self.rotor.evaluate(
                [U], [Omega], [pitch], coefficient=False)
            aeloads[:, 0, i], aeloads[:, 1, i] = self.rotor.distributedAeroLoads(
                U, Omega, pitch, azimuth=90.0)
            #aeloads[:, 0, i] = Np[i]
            #aeloads[:, 1, i] = Tp[i]
            tsr[i] = Omega / 60. * 2 * np.pi * self.Rtip / U

            pitch_init = abs(pitch)
            pitch_angles[i] = pitch

        plot = False
        if plot:
            plt.figure()
            plt.plot(U_oper, pitch_angles, label='CCblade')
            plt.plot(U_oper, Omega_oper, label='CCblade')
            plt.plot(U_oper, cp, label='CCblade')
            #plt.plot(TSR_excel, CP_excel, label='Reference simulated')
            plt.xlabel('$U$')
            plt.ylabel('$\beta$')
            plt.legend(loc='best')
            plt.grid()

        return pitch_angles, cp, aeloads, tsr, Omega_oper, P

    def evaluate_omega(self):
        print 'check'
        # first calculate correct CP by a lot of ccblade runs
        tsr = np.linspace(8.7, 8.8, 1000)
        Uinf = 6.0 * np.ones_like(tsr)
        Omega = Uinf * 30.0 / np.pi * tsr / self.Rtip
        pitch = np.zeros_like(tsr)
        CP, CT, CQ = self.rotor.evaluate(Uinf, Omega, pitch, coefficient=True)
        correctCP = max(CP)
        print 'check'
        # estimate CP using Newton-Raphson iteration using only a few ccblade
        # runs
        tsr = np.array([8.5, 8.75, 9.0])
        Uinf = 6.0 * np.ones_like(tsr)
        Omega = Uinf * 30.0 / np.pi * tsr / self.Rtip
        pitch = np.zeros_like(tsr)

        CP, CT, CQ = self.rotor.evaluate(Uinf, Omega, pitch, coefficient=True)
        CP_init = CP[1]
        CP_init2 = CP[0]
        CP_init3 = CP[2]
        tsr_init = tsr[1]
        tsr_init2 = tsr[0]
        tsr_init3 = tsr[2]
        tsr = tsr_init
        CP = CP_init

        CP_deriv = (CP_init - CP_init2) / (tsr_init - tsr_init2)
        CP_deriv_old = (CP_init2 - CP_init3) / (tsr_init2 - tsr_init3)
        CP_deriv_two = (CP_deriv - CP_deriv_old) / (tsr_init - tsr_init2)

        step = 0

        # Newton-Raphson iteration method according to
        # http://www.cup.uni-muenchen.de/ch/compchem/geom/nr.html
        # this parameter needs to be fixed to some proper value
        while abs(CP_deriv) >= 0.00005:
            tsr_old = tsr
            CP_old = CP
            CP_deriv_old = CP_deriv
            CP_deriv_two_old = CP_deriv_two

            # 0.05 is to reduce maximum stepsize (under relaxation factor)
            # script would diverge without
            tsr = tsr_old - 0.1 * (CP_deriv_old / CP_deriv_two_old)

            Omega = Uinf[0] * 30.0 / np.pi * tsr / self.Rtip

            CP, CT, CQ = self.rotor.evaluate(
                [Uinf[0]], [Omega], [0.0], coefficient=True)
            CP = CP[0]

            CP_deriv = (CP - CP_old) / (tsr - tsr_old)  # backward difference
            # backward difference
            CP_deriv_two = (CP_deriv - CP_deriv_old) / (tsr - tsr_old)

            CP_diff = CP - CP_old

            print step
            print tsr
            print Omega
            print CP_diff
            print CP
            step = step + 1

        if Omega < 5.0:
            Omega = 5.0
        if Omega > 10.0:
            Omega = 10.0

        print Omega
        print 'CP correct is:'
        print correctCP
        print 'CP difference is:'
        print (CP - correctCP)
        print 'CP relative difference is:'
        print 100 * (CP - correctCP) / correctCP


if __name__ == '__main__':

    cfg = {}
    cfg['path_airfoils'] = '/home/becmax/iwes-gitlab/bdtprojects/smartblades/btc_design_model/data/turbine/aero/geo_BlendedAirfoils_export'

    prob = Problem()
    root = prob.root = Group()
    #root.add('comp1', ReadAirfoils(cfg), promotes=['af_data'])
    root.add('comp2', CCBladeEvaluator(), promotes=['af_data'])
    prob.setup()

    print("num connections:", len(prob.root.connections))
    print("num unknowns:", len(prob.root._unknowns_dict),
          "size:", prob.root.unknowns.vec.size)
    print("num params:", len(prob.root._params_dict),
          "size:", prob.root.params.vec.size)

    prob.run()
