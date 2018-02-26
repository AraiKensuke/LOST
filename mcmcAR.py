class mcmcAR:
    #  Simulation params
    processes     = 1
    setname       = None
    env_dirname   = None
    datafn        = None
    rs            = -1
    bFixF         = False
    bFixH         = False
    burn          = None;    NMC           = None
    t0            = None;    t1            = None
    useTrials     = None;    restarts      = 0

    #  Description of model
    model         = None
    rn            = None    #  used for count data
    k             = None

    Bsmpx         = None
    smp_q2        = None
    smp_x00       = None

    dt            = None

    y             = None
    kp            = None

    x             = None   #  true latent state

    q2            = None;    q20           = None
    us             = None;

    smpx          = None
    ws            = None
    x00           = None
    V00           = None

    #  
    _d            = None

    ##### PRIORS
    #  q2  --  Inverse Gamma prior
    #a_q2         = 1e-1;          B_q2         = 1e-6
    #a_q2         = 1e-1;          B_q2         = 1e-11
    a_q2         = -1;          B_q2         = 0
    #  initial states
    u_x00        = None;          s2_x00       = None
    #  u   --  Gaussian prior
    u_u          = -3;             s2_u         = 2.5
    #  initial states
    u_x00        = None;          s2_x00       = None
