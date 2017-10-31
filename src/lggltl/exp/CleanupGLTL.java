package lggltl.exp;

import burlap.behavior.policy.Policy;
import burlap.behavior.policy.PolicyUtils;
import burlap.behavior.singleagent.Episode;
import burlap.behavior.singleagent.auxiliary.EpisodeSequenceVisualizer;
import burlap.behavior.singleagent.planning.Planner;
import burlap.behavior.singleagent.planning.stochastic.rtdp.BoundedRTDP;
import burlap.behavior.singleagent.planning.stochastic.valueiteration.ValueIteration;
import burlap.behavior.valuefunction.ConstantValueFunction;
import burlap.mdp.core.Domain;
import burlap.mdp.core.TerminalFunction;
import burlap.mdp.core.oo.propositional.GroundedProp;
import burlap.mdp.core.oo.propositional.PropositionalFunction;
import burlap.mdp.core.oo.state.OOState;
import burlap.mdp.core.state.State;
import burlap.mdp.singleagent.SADomain;
import burlap.mdp.singleagent.environment.Environment;
import burlap.mdp.singleagent.environment.SimulatedEnvironment;
import burlap.mdp.singleagent.model.FactoredModel;
import burlap.mdp.singleagent.model.RewardFunction;
import burlap.mdp.singleagent.oo.OOSADomain;
import burlap.statehashing.HashableStateFactory;
import burlap.statehashing.simple.SimpleHashableStateFactory;
import burlap.visualizer.Visualizer;
import lggltl.cleanup.CleanupDomain;
import lggltl.cleanup.CleanupVisualiser;
import lggltl.cleanup.state.CleanupState;
import lggltl.gltl.GLTLCompiler;
import lggltl.gltl.state.GLTLState;
import org.yaml.snakeyaml.Yaml;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

/**
 * Created by dilip on 10/26/17.
 */
public class CleanupGLTL {

    public static void main(String[] args) {

        String formula;

        formula = "G4F4&CF1B"; //Always eventually (Green and eventually Blue)
//        formula = "G4F4&RF4B"; //Always eventually (Red and eventually Blue)

        if (args.length > 0) {
            formula = args[0];
        }

        System.out.println("Running for formula " + formula);

        final OOSADomain envDomain;

        CleanupDomain dgen = new CleanupDomain();
        dgen.includeDirectionAttribute(true);
        dgen.includePullAction(true);
        dgen.includeWallPF_s(true);
        dgen.includeLockableDoors(false);
        dgen.setLockProbability(0.0);
        envDomain = dgen.generateDomain();

        State s = CleanupDomain.getClassicState(true);


        //let our GLTL symbol "o" correspond to evaluating whether the agent is in an orange location (a parameterless propositional function)
        // and "b" be whether the agent is in a blue location
        Map<String, GroundedProp> symbolMap = new HashMap<>(1);
        //System.out.println(envDomain.getPropFunctions());
        PropositionalFunction a2r_pf = new CleanupDomain.PF_InRegion(CleanupDomain.PF_AGENT_IN_ROOM,
                new String[]{CleanupDomain.CLASS_AGENT, CleanupDomain.CLASS_ROOM}, false);

        PropositionalFunction b2r_pf = new CleanupDomain.PF_InRegion(CleanupDomain.PF_BLOCK_IN_ROOM,
                new String[]{CleanupDomain.CLASS_BLOCK, CleanupDomain.CLASS_ROOM}, false);


        GroundedProp gp =  new GroundedProp(a2r_pf, new String[]{"agent0", "room1"});
        symbolMap.put("C", gp);
        gp =  new GroundedProp(a2r_pf, new String[]{"agent0", "room0"});
        symbolMap.put("R", gp);
        gp =  new GroundedProp(a2r_pf, new String[]{"agent0", "room2"});
        symbolMap.put("B", gp);

        GLTLCompiler compiler = new GLTLCompiler(formula, symbolMap, envDomain);
        Domain compiledDomain = compiler.generateDomain();
        RewardFunction rf = compiler.generateRewardFunction();
        TerminalFunction tf = compiler.generateTerminalFunction();

        State initialCompiledState = compiler.addInitialTaskStateToEnvironmentState((OOState) s);
        // System.out.println(initialCompiledState.getCompleteStateDescription());

        HashableStateFactory hashingFactory = new SimpleHashableStateFactory();

        //begin planning in our compiled domain
//        Planner planner = new ValueIteration((SADomain) compiledDomain, 1.0, hashingFactory, 0.0000001, 1000);
        Planner planner = new BoundedRTDP((SADomain) compiledDomain, 1.0, hashingFactory, new ConstantValueFunction(0.0), new ConstantValueFunction(1.0),0.0000001, 1000);
//        ((ValueIteration) planner).toggleReachabiltiyTerminalStatePruning(true);
        long startTime = System.nanoTime();
        Policy p = planner.planFromState(initialCompiledState);
        long endTime = System.nanoTime();

        long duration = (endTime - startTime) / 1000000000;
        System.out.println("BRTDP took " + duration + " seconds");

        Environment env = new SimulatedEnvironment((SADomain)compiledDomain,initialCompiledState);

        Episode ea = PolicyUtils.rollout(p, env);

        Episode cw_episode= new Episode(((GLTLState)ea.stateSequence.get(0)).envState);

        for(int count = 0;count<ea.numActions();count++){
//            GridWorldState gs = (GridWorldState)((GLTLState)ea.stateSequence.get(count)).envState;
            CleanupState gsNew = (CleanupState) ((GLTLState)ea.stateSequence.get(count+1)).envState;


//            EnvironmentOutcome eo = new EnvironmentOutcome(gs,ea.actionSequence.get(count),gsNew,0,false);
            cw_episode.transition(ea.actionSequence.get(count),gsNew,0);
//            System.out.println("_____________________");
        }

        Visualizer v = CleanupVisualiser.getVisualizer("data/resources/robotImages");
        //		System.out.println(ea.getState(0).toString());
        new EpisodeSequenceVisualizer(v, envDomain, Arrays.asList(cw_episode));
    }
}
