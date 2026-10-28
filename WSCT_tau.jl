#There are 2 possible properties (Color, Shape). 1 Central card Cc and 2 Oriented cards Cr, Cl. Given a rule (Color or Shape) the correct answer correspond to touch the oriented card with the same property (in rule) of Cc. The rule can change at each trial based on a probability P. If you make a mistake, an error signal is received.
# ====================================================================== #
using LinearAlgebra, Random, Statistics, OrdinaryDiffEq, DifferentialEquations, Interpolations, Plots, MLJ
include("../../JuliaLibrary/RC.jl")

# ====================================================================== #
## Auxiliary Functions
function Opposite_index(rule::Int64) # 1->2, 2->1
    if(rule == 1)
        return 2
    else
        return 1
    end
end

function ComputeCorrectResponse(central::Vector{Int64}, card1::Vector{Int64}, rule::Int64)
    if(central[rule] == card1[rule])
        return -1 #card1 /right
    else
        return +1 #card2
    end
end

# the card state (C,S) is mapped to a unique index as ndx = (C-1)*M[2]+S
function GetCardCode(CardState::Vector{Int64}, State2Max)
    return (CardState[1] -1) * State2Max + CardState[2]
end

# smoothed square wave
function SmoothedSquare(L::Int64, C::Int64)
    v = 0.5*(cos.(range(-π,0.0, length=C)) .+ 1.0);
    r = ones(L);
    r[1:C] = v;
    r[end-C+1:end] = -v .+ 1.0;
    return r;
end

# Plot Network Stimulation in a range of time
function PlotNuExt(nuext::Matrix{Float64}, steps_range::Tuple{Int64, Int64}, n_cards::Int64, CONTEXT_OUT::Bool)
    cards_color = palette(:roma, n_cards);
    error_color = RGB(0.8,0.0,0.2);
    response_color = RGB(0.2,0.2,0.2);
    range_step = steps_range[1]:steps_range[end];
    p = plot(size=(600,350))
    for k=1:n_cards
        plot!(nuext[range_step,k] .+ k*2.1, label="Card $k", color=cards_color[k]);
    end
    plot!(nuext[range_step,end-(1+CONTEXT_OUT)] .+ ((n_cards+1)*2.4+2.2), label="Error Signal", color=error_color);
    if CONTEXT_OUT
        plot!(nuext[range_step,end-1] .+ ((n_cards+1)*2.4+4.6), label="Rule Signal", color=RGB(0.65,0.6,0.8), line=:dash);
    end
    plot!(nuext[range_step,end] .+ (n_cards+1)*2.4, label="Optimal Response", color=response_color, linestyle=:dash);
    plot!(yticks=[], xlabel="Steps", ylabel="Signals", legend=:outerbottom, legendcolumns=4, bottom_margin = 3Plots.mm,left_margin = 5Plots.mm);
    return p
end

#--

mutable struct SignalDuration
    central::Float64
    oriented::Float64
    error::Float64
end

mutable struct SignalSteps
    central::Int64
    oriented::Int64
    error::Int64
end

mutable struct WaitingTime
    central_to_oriented::Float64
    response_delay::Float64 #the response start delayed respect to oriented cards and ends with them
    oriented_to_error::Float64
    inter_trials::Float64
end

mutable struct WaitingSteps
    central_to_oriented::Int64
    response_delay::Int64 #the response start delayed respect to oriented cards and ends with them
    oriented_to_error::Int64
    inter_trials::Int64
end

mutable struct TrialSignal
    central::Vector{Float64}
    oriented::Matrix{Float64}
    error::Vector{Float64}
    response::Vector{Float64}
end
# ====================================================================== #
##
M = [2,2]; # number of possible colors, shapes (= max index)
NCards = M[1] * M[2]; # number of possible cards


NTrialsM, NTrialsL, NTrialsT = 20, 80, 100;
NTrials = NTrialsM + NTrialsL + NTrialsT; # n. trials
central_cards = [[rand(1:M[1]),rand(1:M[2])] for k=1:NTrials];

# simulate rule dynamics
change_prob = 0.2; # change rule probability
error_magnitude = 1.0;
ERROR_SIGN = false;
CONTEXT_OUT = true;
change_rule_ndx = []; # trials when the rule is changed
rules_signed = ones(Int64, NTrials);
for k=2:NTrials
    if rand() < change_prob
        rules_signed[k:end] = -rules_signed[k:end];
        push!(change_rule_ndx, k)
    end
end
rules = Int.(0.5  * rules_signed .+ 1.5);

# initialize the oriented cards to random properties, then we will change them based on picked correct card.
oriented_cards = [[[rand(1:M[1]),rand(1:M[2])],[rand(1:M[1]),rand(1:M[2])]] for k=1:NTrials];
correct_cards = rand(1:2, NTrials); # the right or left one
correct_responses = Int.(2.0*(correct_cards .- 1.5));
# response associated to Learning Trials
# the response must be always correct exept for the change rule moment
optimal_responses = copy(correct_responses);

change_counter = 1;
for k=eachindex(correct_cards) #hence for each trial

    # set the rule property of the correct card equal to that of the central card
    oriented_cards[k][correct_cards[k]][rules[k]] = central_cards[k][rules[k]];

    # set the rule property of the other card different from that of the central card
    if(central_cards[k][rules[k]] == 1)
        oriented_cards[k][Opposite_index(correct_cards[k])][rules[k]] = rand(2:M[rules[k]]);
    elseif(central_cards[k][rules[k]] == M[rules[k]])
        oriented_cards[k][Opposite_index(correct_cards[k])][rules[k]] = rand(1:(M[rules[k]]-1));
    else
        oriented_cards[k][Opposite_index(correct_cards[k])][rules[k]] = rand([1:(central_cards[k][rules[k]]-1);(central_cards[k][rules[k]]+1):M[rules[k]]]);
    end

    # set the other property of the other card equal to that of the correct card (such that the rule cannot be infered from the oriented cards)
    opposite_rule = Opposite_index(rules[k]);
    oriented_cards[k][Opposite_index(correct_cards[k])][opposite_rule] = central_cards[k][opposite_rule];

    # set the other property of the correct card different from that of the correct card
    if(central_cards[k][opposite_rule] == 1)
        oriented_cards[k][correct_cards[k]][opposite_rule] = rand(2:M[opposite_rule]);
    elseif(central_cards[k][opposite_rule] == M[opposite_rule])
        oriented_cards[k][correct_cards[k]][opposite_rule] = rand(1:(M[opposite_rule]-1));
    else
        oriented_cards[k][correct_cards[k]][opposite_rule] = rand([1:(central_cards[k][opposite_rule]-1);(central_cards[k][opposite_rule]+1):M[opposite_rule]]);
    end

    # compute the response with the previous rule if a rule change just happened.
    if(change_counter <= length(change_rule_ndx))
        if(k == change_rule_ndx[change_counter])
            optimal_responses[k] = ComputeCorrectResponse(central_cards[k], oriented_cards[k][1], rules[k-1]);
            change_counter += 1;
        end
    end

end

# error signal associated to Learning Trials (active or not)
optimal_error = error_magnitude * abs.(0.5*(correct_responses - optimal_responses));

## Cards code format: Define the index of JExt to be activated at each trials
central_cards_index = zeros(Int, NTrials);
oriented_cards_index = zeros(Int, 2,NTrials);
for k=eachindex(central_cards_index) #hence for each trial
    central_cards_index[k] = GetCardCode(central_cards[k], M[2]);
    oriented_cards_index[1,k] = GetCardCode(oriented_cards[k][1], M[2]);
    oriented_cards_index[2,k] = GetCardCode(oriented_cards[k][2], M[2]);
end


## Network parameters
N = 500; #300
g = 0.2; #0.1
#τs = 0.001;
τs = 0.001*Vector(LinRange(10,50,N));
#mτ = mean(τs);
mτ = minimum(τs);
dt = mτ/10.;

## =====================================================================
# the time scale is defined by Network mτ
smoothed_steps = round(Int,1.0mτ/dt);
smoothed_steps_context = 5*smoothed_steps;
signal_duration = SignalDuration(9.0mτ, 9.0mτ, 6.0mτ);
signal_steps = SignalSteps(round(Int,signal_duration.central/dt), round(Int,signal_duration.oriented/dt), round(Int,signal_duration.error/dt));

waiting_time = WaitingTime(1.0mτ, 3.0mτ, 1.0mτ, 10.0mτ);
waiting_steps = WaitingSteps(round(Int,waiting_time.central_to_oriented/dt), round(Int,waiting_time.response_delay/dt), round(Int,waiting_time.oriented_to_error/dt), round(Int,waiting_time.inter_trials/dt));

trial_duration = (signal_duration.central +
waiting_time.central_to_oriented +
signal_duration.oriented +
waiting_time.oriented_to_error +
signal_duration.error +
waiting_time.inter_trials);

total_duration = NTrials * trial_duration;
total_steps = round(Int,total_duration/dt);

## =====================================================================
# Signals Shape (Single Trial)

trial_steps = round(Int,trial_duration/dt);
trial_signal = TrialSignal(zeros(trial_steps), zeros(2,trial_steps), zeros(trial_steps), zeros(trial_steps));

# Central card signal
t0_central = 1;
t1_central = signal_steps.central;
trial_signal.central[t0_central:t1_central] .= SmoothedSquare(t1_central - t0_central +1, smoothed_steps);

# Oriented cards signals
t0_oriented = 1 + signal_steps.central + waiting_steps.central_to_oriented;
t1_oriented = t0_oriented + signal_steps.oriented -1;
trial_signal.oriented[1,t0_oriented:t1_oriented] .= -SmoothedSquare(t1_oriented - t0_oriented +1, smoothed_steps); #right
trial_signal.oriented[2,t0_oriented:t1_oriented] .= SmoothedSquare(t1_oriented - t0_oriented +1, smoothed_steps); #left

# Response signal
t0_response = t0_oriented + waiting_steps.response_delay;
t1_response = t1_oriented;
trial_signal.response[t0_response:t1_response] .= SmoothedSquare(t1_response - t0_response +1, smoothed_steps);

# Error card signal
t0_error = t1_oriented + waiting_steps.oriented_to_error;
t1_error = t0_error + signal_steps.error;
trial_signal.error[t0_error:t1_error] .= error_magnitude*SmoothedSquare(t1_error - t0_error +1, smoothed_steps);

##
plot(trial_signal.central, label="central", lw=2);
plot!(trial_signal.oriented[1,:], label="right card", lw=2);
plot!(trial_signal.oriented[2,:], label="left card", lw=2);
plot!(trial_signal.response, label="response", lw=2);
plot!(trial_signal.error, label="error", lw=2);
display(plot!(size=(500,350), title="Trial Prototype", xlabel="steps", ylabel="signal"))

##
tspanM = (dt, NTrialsM * trial_duration);
tspanL = (tspanM[end]+dt, tspanM[end] + NTrialsL * trial_duration);
tspanT = (tspanL[end]+dt, tspanL[end] + NTrialsT * trial_duration);

tspanM_steps = (1, trial_steps*NTrialsM);
tspanL_steps = (tspanM_steps[end]+1, tspanM_steps[end]+trial_steps*NTrialsL);
tspanT_steps = (tspanL_steps[end]+1, tspanL_steps[end]+trial_steps*NTrialsT);

## The network has a number of inputs equals to: Number of possible cards + 1 error channel, and 1 or 2 Output: response channel + context channel
Input(x) = zeros(NCards+1);
Output(x) = zeros(1+CONTEXT_OUT);

## Define Network Stimulation
nuext = zeros(total_steps,NCards+2+CONTEXT_OUT);
correct_ext = zeros(total_steps,1);
rule_ext = zeros(total_steps,1);
trial_step_start = ones(Int,NTrials);
trial_step_end = ones(Int,NTrials);

# to fit central card properties for post learning analysis
central_ctx_nuext = zeros(total_steps,2);
trial_central_ctx_signal = zeros(trial_steps);
trial_central_ctx_signal[t0_central:t1_response] .= SmoothedSquare(t1_response - t0_central +1, smoothed_steps);


@inbounds for k=1:NTrials
    trial_step_end[k] = trial_step_start[k]+trial_steps-1;
    steps_range = trial_step_start[k]:trial_step_end[k];
    
    nuext[steps_range, central_cards_index[k]] += trial_signal.central;

    central_ctx_nuext[steps_range, 1] += trial_central_ctx_signal * sign(central_cards[k][1] - 1.5);
    central_ctx_nuext[steps_range, 2] += trial_central_ctx_signal * sign(central_cards[k][2] - 1.5);

    nuext[steps_range, oriented_cards_index[1,k]] += trial_signal.oriented[1,:];
    nuext[steps_range, oriented_cards_index[2,k]] += trial_signal.oriented[2,:];
    # During Learning Time the error signal is already determined
    if(k <= (NTrialsL+NTrialsM))
        if ERROR_SIGN
            nuext[steps_range, end-(1+CONTEXT_OUT)] .= rules_signed[k] * optimal_error[k] * trial_signal.error;#rule error signal
        else
            nuext[steps_range, end-(1+CONTEXT_OUT)] .= optimal_error[k] * trial_signal.error;#error signal
        end

        
    end

    if CONTEXT_OUT

        if(optimal_error[k] == 1.0) #change rule when the error comes
            before_error_range = trial_step_start[k]:(trial_step_start[k]+t0_error);
            nuext[before_error_range, end-1] .= rules_signed[k-1];
            
            smoothing_error_range = (trial_step_start[k]+t0_error+1):(trial_step_start[k]+t0_error+smoothed_steps_context);
            nuext[smoothing_error_range, end-1] .= (SmoothedSquare(smoothed_steps_context, smoothed_steps_context)*2.0 .- 1.0)*rules_signed[k-1];

            after_error_range = (trial_step_start[k]+t0_error+smoothed_steps_context):trial_step_end[k];
            nuext[after_error_range, end-1] .= rules_signed[k];

        else
            nuext[steps_range, end-1] .= rules_signed[k];
        end
        
    end

    nuext[steps_range, end] .= optimal_responses[k] * trial_signal.response;#optimal response
    correct_ext[steps_range] .= correct_responses[k] * trial_signal.response;#correct response
    rule_ext[steps_range] .= rules[k] * trial_signal.response;#rule

    if(k != NTrials)
        trial_step_start[k+1] = trial_step_start[k]+trial_steps;
    end
end


## Plot Learning session
p = PlotNuExt(nuext, tspanL_steps, NCards, CONTEXT_OUT);
display(plot!(size=(750,400),title="Learning Period Input"))

## Zoom in
p = PlotNuExt(nuext, (tspanL_steps[1], tspanL_steps[1]+trial_steps), NCards, CONTEXT_OUT);
display(plot!(size=(300,400),title="Zoom-in Inputs", legend=false))

##
noiseforLearning = 0.01; #0.01
gextError = 0.5; #0.5
gextCards = 0.1; #0.1
gbackContext = 0.05; # 0.015
gbackResponse = 0.0;
NO_BACK_CONTEXT = false; # don't close loop for context channel
NO_SPARSITY = true;
## =====NETWORK==========
Net = ESN.SetNet(;
netSize=N,
τ = τs,
distribution="Gaussian",
extraInputFunc=Input,
outputFunc=Output,
g=g,
σ_noise=0.0
);

if NO_SPARSITY
    Net.J = randn(N,N);
    Net.J = g*Net.J / maximum(abs.(eigvals(Net.J)));
end

##
PLOT_ACTIVITY = false;
m_gds = 0.6;
Net.J[diagind(Net.J)] .= m_gds;

##
ESN.PlotSpectrum(Net.J)

##
Net.Jext[:, :] .= 0.0;

# cards channel
Net.Jext[:, 1:NCards] = gextCards*randn(N, NCards);

# error channel
Net.Jext[:, end-(1+CONTEXT_OUT)] = gextError*randn(N);

# context channel
if CONTEXT_OUT
    Net.Jext[:, end-CONTEXT_OUT] = gbackContext*randn(N);
end

#response channel
Net.Jext[:, end] = gbackResponse*randn(N);

## Transient
Net.ν .= 0.0;
_ = ESN.EulerExplicit(Net, tspanM; dt=dt, nuext=nuext[tspanM_steps[1]:tspanM_steps[end], :]);

## Learning
nuextL = nuext[tspanL_steps[1]:tspanL_steps[end], :];
tL, νL = ESN.EulerExplicit(Net, tspanL; dt=dt, dtsave=dt, nuext=nuextL);

##
if PLOT_ACTIVITY
    display(plot(size=(400,200),νL[1:50:end,1:10:end]', ylims=(-1.05, 1.05),
        legend=false, xlabel="Time", ylabel="ν",title="Learning State"))
end

##
ResponseOrigL = transpose(nuext[tspanL_steps[1]:tspanL_steps[end], end]);
ResponseOrigT = transpose(nuext[tspanT_steps[1]:tspanT_steps[end], end]);
OutOrigT = transpose(nuext[tspanT_steps[1]:tspanT_steps[end], end-CONTEXT_OUT:end]);
OutOrigL = transpose(nuext[tspanL_steps[1]:tspanL_steps[end], end-CONTEXT_OUT:end]);
Wout = ESN.LinearMap(νL, OutOrigL, noiseforLearning);

NetC = ESN.Copy_Net(Net);
if NO_BACK_CONTEXT
    # feedback just on response channel
    NetC = ESN.LinearClosingLoop(Net, Wout[1+CONTEXT_OUT,:]');
else
    NetC = ESN.LinearClosingLoop(Net, Wout);
end

OutL = Wout*νL;
ResponseL = OutL[end,:];
## Fit Check
plot(ResponseOrigL[:], label="Optimal Response");
plot!(ResponseL, label="Fitted Response",legend=true, size=(700,300), xlabel="Steps", title="Learning Check", bottom_margin=5Plots.mm)

## Context Fit Check
OutContextL = [];
if CONTEXT_OUT
    OutContextL = OutOrigL[end-CONTEXT_OUT,:];
    plot(OutContextL, label="Real Context");
    plot!(OutL[end-CONTEXT_OUT,:], label="Fitted Context",legend=true, size=(700,300), xlabel="Steps", title="Learning Check", bottom_margin=5Plots.mm)
end
## Testing
NetC.ν = Net.ν;
tT = tspanT[1]:dt:tspanT[end];
νT = zeros(N, length(tT));
ResponseT = zeros(length(tT));

NT0 = NTrialsM+NTrialsL;
nuext[trial_step_start[NT0+1]:end,end-(1+CONTEXT_OUT)] .= 0.0; # no precomputed error signal in test
response_steps = t1_response-t0_response;
fake_span_pre = (dt,t1_response*dt);
fake_span_post = (dt,(trial_steps-t1_response)*dt);

nuext_single_trial = zeros(trial_steps,size(nuext,2));
net_out_accordance_optimal = zeros(NTrialsT);
net_out_accordance_correct = zeros(NTrialsT);
optimal_accordance_correct = zeros(NTrialsT);

for k=1:NTrialsT
    nuext_single_trial = nuext[trial_step_start[NT0+k]:trial_step_end[NT0+k],:];
    pre_range = ((k-1)*trial_steps+1):(t1_response+(k-1)*trial_steps);
    post_range = (pre_range[end]+1):trial_steps*k;

    # up to network response
    _, νT[:,pre_range] = ESN.EulerExplicit(NetC, fake_span_pre; dt=dt, dtsave = dt, nuext=nuext_single_trial[t0_central:t1_response,:]);

    ResponseT[pre_range[end-response_steps:end]] = Wout[end,:]'*νT[:,pre_range[end-response_steps:end]];
    net_response = sign(sum(ResponseT[pre_range[end-response_steps:end]]));
    net_out_accordance_optimal[k] = Int(optimal_responses[NT0+k] == net_response);
    net_out_accordance_correct[k] = Int(correct_responses[NT0+k] == net_response);
    optimal_accordance_correct[k] = Int(correct_responses[NT0+k] == optimal_responses[NT0+k]);
    
    # error stimulation
    if(net_out_accordance_correct[k] == 0.0)
        if ERROR_SIGN
            nuext_single_trial[:, end-(1+CONTEXT_OUT)] .= rules_signed[NT0+k]*trial_signal.error;
            nuext[trial_step_start[NT0+k]:trial_step_end[NT0+k],end-(1+CONTEXT_OUT)] .= rules_signed[NT0+k]*trial_signal.error;
        else
            nuext_single_trial[:, end-(1+CONTEXT_OUT)] .= trial_signal.error;
            nuext[trial_step_start[NT0+k]:trial_step_end[NT0+k],end-(1+CONTEXT_OUT)] .= trial_signal.error;
        end
    end

    _, νT[:,post_range] = ESN.EulerExplicit(NetC, fake_span_post; dt=dt, dtsave = dt, nuext=nuext_single_trial[(t1_response+1):end,:]);

end

nuextT = nuext[tspanT_steps[1]:tspanT_steps[end], :];
##
accuracy_from_correct = mean(net_out_accordance_correct);
accuracy_from_optimal = mean(net_out_accordance_optimal);
optimal_accuracy = mean(optimal_accordance_correct);

println("Accuracy: $(accuracy_from_correct*100)%\nAccordance with Optimal: $(accuracy_from_optimal*100)%\nOptimal Accuracy: $(optimal_accuracy*100)%")

## plot testing state
if PLOT_ACTIVITY
    plot(size=(400,200),tT, νT[1:50:end,:]', ylims=(-1.05, 1.05),
    legend=false, xlabel="Time", ylabel="ν",title="Testing State")
end

##
OutT = Wout*νT;

## Context, Testing
OutContextT = [];
OutOrigContextT = [];
if CONTEXT_OUT
    OutOrigContextT = OutOrigT[end-CONTEXT_OUT,:];
    OutContextT = OutT[end-CONTEXT_OUT,:];
    plot(OutOrigContextT, label="Real Context");
    plot!(OutContextT, label="Recognized Context",legend=true, size=(700,300), xlabel="Steps", title="Testing", bottom_margin=5Plots.mm)
end


## Test Check
Red = RGBA(1,0.4,0.6,1);
Blue = RGBA(0.2,0.5,0.7,1);
Grey = RGBA(0.5,0.5,0.5,0.5);

plot(ResponseOrigT', label="Optimal Response", color=Grey, lw=1.5);
#plot!(correct_ext[tspanT_steps[1]:tspanT_steps[end]], linestyle=:dash, lw=1.5, color=:red, label="Correct Response")
plot!(ResponseT, label="Autonomous Response",legend=true, size=(700,300), xlabel="Steps", title="Testing Check", bottom_margin=5Plots.mm, color=Blue)

## Plot Task
p = PlotNuExt(nuext, tspanT_steps, NCards, CONTEXT_OUT);
annotate!(length(tT)/2, -6.0, text("Accuracy: $(accuracy_from_correct*100)%\nAccordance with Optimal: $(accuracy_from_optimal*100)%\nOptimal Accuracy: $(optimal_accuracy*100)%", :black, :center, 10));
#plot!(correct_ext[tspanT_steps[1]:tspanT_steps[end]] .+ (NCards+1)*2.4, linestyle=:dash, lw=1.5, color=:red, label="Correct Response");
if !CONTEXT_OUT
    plot!(rule_ext[tspanT_steps[1]:tspanT_steps[end]] .+ ((NCards+1)*2.4 + 5.0), lw=1.5, color=:black, label="Rule");
else
    plot!(OutContextT .+ ((NCards+1)*2.4+4.6), label="Net Context", color=RGB(0.5,0.8,0.4));
end
plot!(size=(1000,650), ResponseT .+ (NCards+1)*2.4, color=Blue, label="Net Response", title="Network Behaviour during Test", left_margin=8Plots.mm, bottom_margin=15Plots.mm)

## Plot Task Zoom-in
zoom_l = 5000;
p = PlotNuExt(nuext, (tspanT_steps[1], tspanT_steps[1]+zoom_l), NCards, CONTEXT_OUT);
annotate!(zoom_l/2, -6.0, text("Accuracy: $(accuracy_from_correct*100)%\nAccordance with Optimal: $(accuracy_from_optimal*100)%\nOptimal Accuracy: $(optimal_accuracy*100)%", :black, :center, 10));
#plot!(correct_ext[tspanT_steps[1]:tspanT_steps[end]] .+ (NCards+1)*2.4, linestyle=:dash, lw=1.5, color=:red, label="Correct Response");
if !CONTEXT_OUT
    plot!(rule_ext[tspanT_steps[1]:tspanT_steps[1]+zoom_l] .+ ((NCards+1)*2.4 + 5.0), lw=1.5, color=:black, label="Rule");
else
    plot!(OutContextT[1:zoom_l] .+ ((NCards+1)*2.4+4.6), label="Net Context", color=RGB(0.5,0.8,0.4));
end
plot!(size=(1000,650), ResponseT[1:zoom_l] .+ (NCards+1)*2.4, color=Blue, label="Net Response", title="Little snapshot Network Behaviour during Test", left_margin=8Plots.mm, bottom_margin=15Plots.mm)


## s distribution after learning
p1 = scatter(NetC.J[diagind(NetC.J)], legend=false, title="post learning", ms=3, alpha=0.2, xlabel="Index", ylabel="self coupling", label="post");
scatter!(Net.J[diagind(Net.J)], legend=true, title="post learning", ms=3, alpha=0.2, xlabel="Index", ylabel="self coupling", label="pre");

p2 = plot([m_gds+0.01, m_gds-0.01, m_gds-0.01, m_gds+0.01, m_gds+0.01],[0.0, 0.0,N, N, 0.0], xlims=(m_gds-0.51,m_gds+0.51), legend=false, line = 1, color=:black, fill = (0, RGB(0.8, 0.88, 0.94)), xlabel="s", ylabel="count", title="s distribution (Pre)");

p3 = histogram(NetC.J[diagind(NetC.J)], legend=false, title="s distribution (Post)", xlabel="s", ylabel="count", color=RGB(0.8, 0.88, 0.94), bins=50);
plot(p1,p2,p3,layout=(1,3), size=(900,300), bottom_margin=5Plots.mm,left_margin=5Plots.mm)

## J correction
ΔJ = (Net.J - NetC.J);
heatmap(ΔJ, title="Couplings variation, |ΔJ|/|J₀|=$(round(norm(ΔJ)/(norm(Net.J)), digits=3))", size=(500,400), cmap=:magma, bottom_margin=2Plots.mm, left_margin=2Plots.mm, right_margin=5Plots.mm, xlabel="index", ylabel="index", colorbar_title="ΔJ", c=:balance, clims=(-maximum(abs.(ΔJ)), maximum(abs.(ΔJ))))

## self coupling weights in context perception and response map
if CONTEXT_OUT
    p1 = plot(τs,abs.(Wout[1,:]), seriestype=:sticks,
    linestyle=:solid, marker=:none, xlabel="τ", ylabel="|W context|", ms=2, color=:deepskyblue4, lw=2.5, alpha=0.7);

    p2 = plot(τs,abs.(Wout[2,:]), seriestype=:sticks,
    linestyle=:solid, marker=:none, xlabel="τ", ylabel="|W response|", ms=2, color=:deepskyblue4, lw=2.5, alpha=0.7);

    plot(p1,p2, layout=(1,2), size=(800,400), legend=false, left_margin=3Plots.mm, bottom_margin=2Plots.mm)
end

## self coupling weights in context perception and response map
if CONTEXT_OUT
    p1 = plot(NetC.J[diagind(NetC.J)],abs.(Wout[1,:]), seriestype=:sticks,
    linestyle=:solid, marker=:none, xlabel="s", ylabel="|W context|", ms=2, color=:deepskyblue4, lw=2.5, alpha=0.7);

    p2 = plot(NetC.J[diagind(NetC.J)],abs.(Wout[2,:]), seriestype=:sticks,
    linestyle=:solid, marker=:none, xlabel="s", ylabel="|W response|", ms=2, color=:deepskyblue4, lw=2.5, alpha=0.7);

    plot(p1,p2, layout=(1,2), size=(800,400), legend=false, left_margin=3Plots.mm, bottom_margin=2Plots.mm)
end


##
function Plot_context_pca(pcs::AbstractMatrix{Float64}, OutContext::Vector{Float64}, error_vec::Vector{Float64}, fsize::Tuple{Int64, Int64}; TH_COLOR::Float64=0.0, PLANE::Bool=false, lims=[], camera=[45.0,45.0,1.0])
    color_c1 = RGB(1.0, 0.3, 0.3);
    color_c2 = RGB(0.2, 0.6, 1.0);
    color_error = RGB(0.1,0.1,0.1);
    color_line = RGB(0.9,0.9,0.9);

    ndx_c1 = findall(OutContext .> TH_COLOR);
    ndx_c2 = findall(OutContext .< -TH_COLOR);
    ndx_error = findall(error_vec .> TH_COLOR);
    
    p1 = plot(pcs[1,:], pcs[2,:], xlabel="PC₁", ylabel="PC₂", color=color_line, label="");
    scatter!(pcs[1,ndx_error], pcs[2,ndx_error], color=:white, markerstrokecolor=color_error,alpha=1.0, ms=4, marker=:circle, label="error sig.");
    scatter!(pcs[1,ndx_c1], pcs[2,ndx_c1], color=color_c1, markerstrokecolor=color_c1, ms=2, alpha=1.0, label="ctx sig. 1");
    scatter!(pcs[1,ndx_c2], pcs[2,ndx_c2], color=color_c2, markerstrokecolor=color_c2, ms=2, alpha=1.0, label="ctx sig. 2");

    p2 = plot(pcs[1,:], pcs[2,:], pcs[3,:], xlabel="PC₁", ylabel="PC₂", zlabel="PC₃", color=color_line, label="");
    scatter!(pcs[1,ndx_error], pcs[2,ndx_error], pcs[3,ndx_error], color=:white, markerstrokecolor=color_error,alpha=1.0, ms=4, marker=:circle, label="error sig.");
    scatter!(pcs[1,ndx_c1], pcs[2,ndx_c1], pcs[3,ndx_c1], color=color_c1, markerstrokecolor=color_c1, ms=1.5, alpha=1.0, label="ctx sig. 1");
    scatter!(pcs[1,ndx_c2], pcs[2,ndx_c2], pcs[3,ndx_c2], color=color_c2, markerstrokecolor=color_c2, ms=1.5, alpha=1.0, label="ctx sig. 2");

    if PLANE
        x = range(-20, 20, length=50);
        y = range(-20, 20, length=50);
        f(x, y) = 0;
        # Create a surface plot with alpha=0.5 for transparency
        surface!(x, y, f, color=RGB(0.9,0.8,0.7), alpha=0.2, cbar=false);
    end

    if !isempty(lims)
        plot!(xlims=lims[1], ylims=lims[2], zlims=lims[3])
    end
    plot!(camera=(camera[1], camera[2], camera[3]));

    plot_layout = @layout [a{0.35w} b{0.65w}];
    p=plot(p1,p2, layout = plot_layout, legend=true, size=fsize, bottom_margin=5Plots.mm, left_margin=5Plots.mm);
    return p
end


function Plot_answ_pca(pcs::AbstractMatrix{Float64}, OutResponse::Vector{Float64}, central_vec::Vector{Float64}, fsize::Tuple{Int64, Int64}; TH_COLOR::Float64=0.0, PLANE::Bool=false, lims=[], camera=[45.0,45.0,1.0])
    color_c1 = RGB(0.2, 0.9, 0.3);
    color_c2 = RGB(0.9, 0.3, 0.9);
    color_central = RGB(0.4,0.1,0.1);
    color_line = RGB(0.9,0.9,0.9);

    ndx_c1 = findall(OutResponse .> TH_COLOR);
    ndx_c2 = findall(OutResponse .< -TH_COLOR);
    ndx_central = findall(central_vec .> TH_COLOR);
    
    p1 = plot(pcs[1,:], pcs[2,:], xlabel="PC₁", ylabel="PC₂", color=color_line, label="");
    scatter!(pcs[1,ndx_central], pcs[2,ndx_central], color=:white, markerstrokecolor=color_central,alpha=1.0, ms=4, marker=:circle, label="central sig.");
    scatter!(pcs[1,ndx_c1], pcs[2,ndx_c1], color=color_c1, markerstrokecolor=color_c1, ms=2, alpha=1.0, label="answ R");
    scatter!(pcs[1,ndx_c2], pcs[2,ndx_c2], color=color_c2, markerstrokecolor=color_c2, ms=2, alpha=1.0, label="answ L");

    p2 = plot(pcs[1,:], pcs[2,:], pcs[3,:], xlabel="PC₁", ylabel="PC₂", zlabel="PC₃", color=color_line, label="");
    scatter!(pcs[1,ndx_central], pcs[2,ndx_central], pcs[3,ndx_central], color=:white, markerstrokecolor=color_central,alpha=1.0, ms=4, marker=:circle, label="central sig.");
    scatter!(pcs[1,ndx_c1], pcs[2,ndx_c1], pcs[3,ndx_c1], color=color_c1, markerstrokecolor=color_c1, ms=1.5, alpha=1.0, label="answ R");
    scatter!(pcs[1,ndx_c2], pcs[2,ndx_c2], pcs[3,ndx_c2], color=color_c2, markerstrokecolor=color_c2, ms=1.5, alpha=1.0, label="answ L");

    if PLANE
        x = range(-20, 20, length=50);
        y = range(-20, 20, length=50);
        f(x, y) = 0;
        # Create a surface plot with alpha=0.5 for transparency
        surface!(x, y, f, color=RGB(0.9,0.8,0.7), alpha=0.2, cbar=false);
    end

    if !isempty(lims)
        plot!(xlims=lims[1], ylims=lims[2], zlims=lims[3])
    end
    plot!(camera=(camera[1], camera[2], camera[3]));

    plot_layout = @layout [a{0.35w} b{0.65w}];
    p=plot(p1,p2, layout = plot_layout, legend=true, size=fsize, bottom_margin=5Plots.mm, left_margin=5Plots.mm);
    return p
end

##
fsize = (900,600);

ΨL, H, p = ENC.Pca(νL);
ED = round(exp(H-1), digits=2);
println("Estimated Embedding Dimension: $ED")
pcs = ΨL'*νL;

if CONTEXT_OUT
    p=Plot_context_pca(pcs, OutContextL, nuextL[:,end-(1+CONTEXT_OUT)], fsize);
    plot!(title="PCA_L on νL, ED=$ED")
else
    p1 = plot(pcs[1,:], pcs[2,:], xlabel="PC₁", ylabel="PC₂");
    p2 = plot(pcs[1,:], pcs[2,:], pcs[3,:], xlabel="PC₁", ylabel="PC₂", zlabel="PC₃");
    plot_layout = @layout [a{0.35w} b{0.65w}];
    plot(p1,p2, layout = plot_layout, legend=false, size=fsize, bottom_margin=5Plots.mm, left_margin=5Plots.mm, title="PCA_L on νL, ED=$ED")
end

##
pcs = ΨL'*νT;

if CONTEXT_OUT
    p = Plot_context_pca(pcs, OutContextT, nuextT[:,end-(1+CONTEXT_OUT)], fsize);
    plot!(title="PCA_L on νT")
else
    p1 = plot(pcs[1,:], pcs[2,:], xlabel="PC₁", ylabel="PC₂");
    p2 = plot(pcs[1,:], pcs[2,:], pcs[3,:], xlabel="PC₁", ylabel="PC₂", zlabel="PC₃");
    plot_layout = @layout [a{0.35w} b{0.65w}];
    plot(p1,p2, layout = plot_layout, legend=false, size=fsize, bottom_margin=5Plots.mm, left_margin=5Plots.mm, title="PCA_L on νT")
end

##
Ψ, H, p = ENC.Pca(νT);
ED = round(exp(H-1), digits=2);
println("Estimated Embedding Dimension: $ED")
pcs = Ψ'*νT;


#camera = [0.0,20.0,1.0];
camera = [70.0,50.0,1.0];
if CONTEXT_OUT
    p = Plot_context_pca(pcs, OutContextT, nuextT[:,end-(1+CONTEXT_OUT)], fsize);
    plot!(title="PCA_T on νT, ED=$ED";camera=camera)
else
    p1 = plot(pcs[1,:], pcs[2,:], xlabel="PC₁", ylabel="PC₂");
    p2 = plot(pcs[1,:], pcs[2,:], pcs[3,:], xlabel="PC₁", ylabel="PC₂", zlabel="PC₃");
    plot_layout = @layout [a{0.35w} b{0.65w}];
    plot(p1,p2, layout = plot_layout, legend=false, size=fsize, bottom_margin=5Plots.mm, left_margin=5Plots.mm, title="PCA_T on νT, ED=$ED")
end

##
pcs_mixed = reduce(hcat,[OutContextT,pcs[1,:],ResponseT])';
Plot_context_pca(pcs_mixed, OutContextT, nuextT[:,end-(1+CONTEXT_OUT)], fsize; PLANE=true, lims=[[-1.5,1.5],[-15,15],[-1,+1]], camera=camera);
plot!(xlabel="W_Ctx", ylabel="PC₁", zlabel="W_Answ", legend=false)

##
W_central = ESN.LinearMap(νL, central_ctx_nuext[tspanL_steps[1]:tspanL_steps[end], :]', noiseforLearning);

central_ctx_origT = central_ctx_nuext[tspanT_steps[1]:tspanT_steps[end], :]';
central_ctx_T = W_central * νT;
println(cor(central_ctx_origT', central_ctx_T'))
##
pcs_mixed = reduce(hcat,[OutContextT,central_ctx_origT[1,:], central_ctx_origT[2,:]])';
Plot_context_pca(pcs_mixed, OutContextT, nuextT[:,end-(1+CONTEXT_OUT)], fsize);
plot!(xlabel="W_Ctx", ylabel="W_C₁", zlabel="W_C₂", legend=false)

##
pcs_mixed = reduce(hcat,[ResponseT,central_ctx_origT[1,:], central_ctx_origT[2,:]])';
#pcs_mixed = reduce(hcat,[ResponseT,central_ctx_origT[1,:], OutContextT])';
Plot_answ_pca(pcs_mixed, ResponseT, sum(nuextT[:,1:NCards], dims=2)[:], fsize; TH_COLOR = 0.25, camera=camera);
plot!(xlabel="W_Answ", ylabel="W_C₁", zlabel="W_C₂", legend=true)

##
pcs_mixed = reduce(hcat,[pcs[1,:],pcs[2,:], pcs[3,:]])';
#pcs_mixed = reduce(hcat,[ResponseT,central_ctx_origT[1,:], OutContextT])';
Plot_answ_pca(pcs_mixed, ResponseT, sum(nuextT[:,1:NCards], dims=2)[:], fsize; TH_COLOR = 0.5, camera=camera);
plot!(xlabel="PC₁", ylabel="PC₂", zlabel="PC₃", legend=false)

##
pcs_mixed = reduce(hcat,[pcs[2,:], pcs[3,:],pcs[1,:]])';
#pcs_mixed = reduce(hcat,[ResponseT,central_ctx_origT[1,:], OutContextT])';
Plot_answ_pca(pcs_mixed, ResponseT, sum(nuextT[:,1:NCards], dims=2)[:], fsize; TH_COLOR = 0.5, camera=camera);
plot!(xlabel="PC₂", ylabel="PC₃", zlabel="PC₁", legend=false)

##
pcs_mixed = reduce(hcat,[pcs[1,:], pcs[3,:],pcs[2,:]])';
#pcs_mixed = reduce(hcat,[ResponseT,central_ctx_origT[1,:], OutContextT])';
Plot_answ_pca(pcs_mixed, ResponseT, sum(nuextT[:,1:NCards], dims=2)[:], fsize; TH_COLOR = 0.5, camera=camera);
plot!(xlabel="PC₁", ylabel="PC₃", zlabel="PC₂", legend=false)


# ========================================================= #
## ------ STM task --------
# ========================================================= #
using Clustering, Distances

function BeatingGenerator(n_bumps::Int, bump_lasting_time::Float64, inter_bump_time::Float64, dt::Float64)
    bump_steps = round(Int,bump_lasting_time/dt);
    inter_steps = round(Int,inter_bump_time/dt);
    total_steps = (bump_steps+inter_steps)*n_bumps;
    nuext=zeros(total_steps, n_bumps);

    tInter = [];
    tIext = [];
    @inbounds for k=1:n_bumps
        t0 = (k-1)*(bump_steps+inter_steps)+1;
        t1 = t0 + bump_steps;
        nuext[t0:t1,k] .= 1.0;

        push!(tInter,t1:t1+inter_steps);
        push!(tIext,t0:t1);
    end
    return tIext,tInter,nuext
end

#----
function STM_Task(NetSTM::ESN.NetType, nuextSTM::AbstractMatrix{Float64}, nuextSTMs::AbstractMatrix{Float64}, tspanSTMs::Tuple{Float64, Float64}, SEQ_TRANS::Int; dtsave::Float64=5*dt)

    ## Transient Period
    tspanSTM = (dt, size(nuextSTM,1)*dt);
    for k=1:SEQ_TRANS
        ESN.EulerExplicit(NetSTM, tspanSTM; dt=dt, nuext=nuextSTM);
    end

    _, νSTM = ESN.EulerExplicit(NetSTM, tspanSTMs; dt=dt, dtsave=dtsave, nuext=nuextSTMs);

    return NetSTM, νSTM
end
#----
function Get_AC(NetSTM::ESN.NetType,νH::AbstractMatrix{Float64}, Samples::Int)
    NP = size(NetSTM.Jext,2);
    States = mean(reshape(νH', Samples, size(νH, 2) ÷ Samples, size(νH, 1)), dims=1)[1,:,:];
    #StatesNum = size(States, 1);
    
    DY = pairwise(CorrDist(), States');
    Z = hclust(DY, linkage=:complete);
    
    T = cutree(Z, k=NP);
    Centroids = zeros(NP, NetSTM.netSize);
    for nc in 1:NP
        Centroids[nc, :] = mean(States[T .== nc, :], dims=1);
    end
    
    CovMat = cor([Centroids; NetSTM.Jext']');
    ndxCS = zeros(Int, NP);
    for nc in 1:NP
        _, ndxCS[nc] = findmax(CovMat[1:NP, NP + nc]);
    end
    Centroids = Centroids[ndxCS, :];
    CovMat = cor([Centroids; NetSTM.Jext']');
    
    AC_temp = zeros(2 * NP - 1);
    AC2_temp = zeros(2 * NP - 1);
    ACsamples_temp = zeros(2 * NP - 1);
    
    R = cor(Centroids');
    ndxMask = 0:NP-1;
    for nr in 1:NP
        AC_temp[ndxMask .+ nr] += R[end - nr + 1, :];
        AC2_temp[ndxMask .+ nr] += R[end - nr + 1, :].^2;
        ACsamples_temp[ndxMask .+ nr] .+= 1;
    end
    AC = AC_temp ./ ACsamples_temp;
    #AC2 = AC2_temp ./ ACsamples_temp;
    ACsd = AC2_temp ./ ACsamples_temp - (AC_temp ./ ACsamples_temp).^2;
    
    AC_stim_temp = zeros(2 * NP - 1);
    ACsamples_stim_temp = zeros(2 * NP - 1);
    
    R_stim = cor(NetSTM.Jext)
    for nr in 1:NP
        AC_stim_temp[ndxMask .+ nr] += R_stim[end - nr + 1, :];
        ACsamples_stim_temp[ndxMask .+ nr] .+= 1;
    end
    AC_stim = AC_stim_temp ./ ACsamples_stim_temp;

    return AC, ACsd, AC_stim
end
##

NP = 6; # n. patterns
SEQ_TRANS = 5; # transient sequence repetitions
SEQ_REP = 30; # sequence repetitions
STIM_INT = 1.0e3; # intensity of the stimulus
bump_time = 5.0mτ; # bump lasting time
inter_time = 5.0mτ; # waiting time
dtsave = 10*dt;

N_NODES = round(Int,0.25*(N*NP)); #(on 25% of nodes)

JextSTM = zeros(N,NP);
for k=1:NP
    JextSTM[rand((1:NetSTM.netSize),N_NODES),k] .= STIM_INT;
end

## -----
#NetSTM = ESN.Copy_Net(Net); # pre training
NetSTM = ESN.Copy_Net(NetC); # post training
NetSTM.ν .= 0.0;
NetSTM.Jext = JextSTM;

inter_time_multiplier = 1.0;
_,_, nuextSTM = BeatingGenerator(NP,bump_time,inter_time*inter_time_multiplier,dt);
nuextSTMs = repeat(nuextSTM,SEQ_REP,1);
tspanSTMs = (dt, size(nuextSTM,1)*SEQ_REP*dt);

@time NetSTM, νSTM = STM_Task(NetSTM, nuextSTM, nuextSTMs, tspanSTMs, SEQ_TRANS; dtsave=dtsave);
AC, ACsd, AC_stim = Get_AC(NetSTM, νSTM, 5);

##
title = "τₛₑₚ=$(round(inter_time*inter_time_multiplier, digits=2)) s";
p1 = bar(-NP .+ 1:NP .- 1, AC, yerr=ACsd,fillcolor=:gray80, xlims=(-0.5 - NP / 2, 0.5 + NP / 2), ylims=(-0.2, 1), xlabel="Centroid distance n", ylabel="Corr. coeff. ρ", title=title, box=:on, grid=:none, size=(250,400), legend=false);

p2 = bar(-NP .+ 1:NP .- 1, AC_stim,fillcolor=:gray80, xlims=(-0.5 - NP / 2, 0.5 + NP / 2), ylims=(-0.2, 1), xlabel="Stim. distance n", ylabel="Corr. coeff. ρ", title=title, box=:on, grid=:none, size=(250,400), legend=false);

plot(p1,p2, size=(500,400))

#plot(νSTM[:,1:25]', legend=false)