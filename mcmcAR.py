class mcmcAR:
    #  Simulation params
    processes     = 1
    setname       = None
    env_dirname   = None
    datafn        = None
    rs            = -1
    bFixF         = False
    bFixH         = False
    ITERS         = None
    minITERS      = 3000
    stationaryDuration  = 4000
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

    y             = None     #  0 1 or counts

    gau_obs       = None     #  after PG variables, y turned into gaussian obs
    gau_var       = None     #  after PG variables, variance of gaussian obs
    kp            = None

    x             = None   #  true latent state

    q2            = None;    q20           = None
    us             = None;

    smpx          = None
    ws            = None
    x00           = None
    V00           = None

    mg_blocksize  = 200
    mg_points     = 20

    N     = None;    #  # of data for each trial
    Ns    = None;    #  # of data for each trial
    TR    = None;    #  # of trials
    #  KF params
    p_x   = None;    p_V   = None;    p_Vi  = None
    f_x   = None;    f_V   = None
    s_x   = None;    s_V   = None
    J     = None
    Ik    = None

    #  params for AR state
    k     = None;
    ks    = None;
    F     = None
    Fs    = None
    Q     = None;
    G     = None

    #  generative and generated data
    x     = None;

    #  original spks
    dN    = None

    f_x           = None
    last_iter     = -1

    ##### PRIORS
    #  q2  --  Inverse Gamma prior
    #a_q2         = 1e-1;          B_q2         = 1e-6
    #a_q2         = 1e-1;          B_q2         = 1e-11
    ##  B / (a+1)
    a_q2         = None;          B_q2         = None
    #a_q2         = 50;          B_q2         = 1e-12
    #  initial states
    u_x00        = None;          s2_x00       = None
    #  u   --  Gaussian prior
    #u_u          = -3;             s2_u         = 2.5
    u_u          = None;             s2_u         = 1e-4
    #  initial states
    u_x00        = None;          s2_x00       = None

    runDir = None
