pe = pyenv(Version="/Users/lancexwq/Ries Lab/LocMoFitPy2/.pixi/envs/default/bin/python", ExecutionMode= "OutOfProcess");

data = readmatrix("./notebooks/data/simulated_data_spcap.csv");
locs = data(:, 1:3);
loc_precisions = data(:,4:6);

tic
bridge_dir = "/Users/lancexwq/Ries Lab/LocMoFitPy2";
if count(py.sys.path, bridge_dir) == 0
    insert(py.sys.path, int32(0), bridge_dir);
end
api = py.importlib.import_module("matlab_api");

kwargs = py.dict;
freeze = py.tuple({});

res = api.run_locmofit( ...
    "SphericalCap", ...
    locs, ...
    loc_precisions, ...
    pyargs( ...
        "seed", int64(3), ...
        "model_init_kwargs", kwargs, ...
        "freeze", freeze, ...
        "max_iter", int64(200), ...
        "tol", 1e-12 ...
    ) ...
);

final_loss = double(res{"final_loss"});
positions  = double(res{"positions"});
params_py  = res{"params"};
toc


ground_truth = readmatrix("./notebooks/data/ground_truth_spcap.csv");
scatter3(positions(:,1), positions(:,2), positions(:,3))
hold on
scatter3(ground_truth(:,1), ground_truth(:,2), ground_truth(:,3))
daspect([1 1 1])
legend('model points','localizations')