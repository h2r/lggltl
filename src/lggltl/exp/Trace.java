package lggltl.exp;

import burlap.behavior.policy.Policy;
import burlap.behavior.policy.PolicyUtils;
import burlap.behavior.singleagent.Episode;
import burlap.behavior.singleagent.planning.Planner;
import burlap.behavior.singleagent.planning.stochastic.valueiteration.ValueIteration;
import burlap.mdp.core.Domain;
import burlap.mdp.core.TerminalFunction;
import burlap.mdp.core.oo.propositional.GroundedProp;
import burlap.mdp.core.oo.propositional.PropositionalFunction;
import burlap.mdp.core.oo.state.OOState;
import burlap.mdp.core.state.State;
import burlap.mdp.singleagent.SADomain;
import burlap.mdp.singleagent.model.RewardFunction;
import burlap.mdp.singleagent.oo.OOSADomain;
import burlap.statehashing.HashableStateFactory;
import burlap.statehashing.simple.SimpleHashableStateFactory;
import lggltl.cleanup.CleanupDomain;
import lggltl.gltl.GLTLCompiler;
import org.yaml.snakeyaml.Yaml;

import java.util.HashMap;
import java.util.Map;

/**
 * Created by dilip on 10/26/17.
 */
public class Trace {

    public static void main(String[] args) {

        String formula;

        formula = "F4R";

        if (args.length > 0) {
            formula = args[0];
        }

        System.out.println("Running for formula " + formula);

        final OOSADomain envDomain;

        CleanupDomain dgen = new CleanupDomain();
        dgen.includeDirectionAttribute(true);
        dgen.includePullAction(true);
        dgen.includeWallPF_s(true);
        dgen.includeLockableDoors(true);
        dgen.setLockProbability(0.5);
        envDomain = dgen.generateDomain();

        State s = CleanupDomain.getClassicState(true);


        //let our GLTL symbol "o" correspond to evaluating whether the agent is in an orange location (a parameterless propositional function)
        // and "b" be whether the agent is in a blue location
        Map<String, GroundedProp> symbolMap = new HashMap<>(1);
        //System.out.println(envDomain.getPropFunctions());
        PropositionalFunction pf = new CleanupDomain.PF_InRegion(CleanupDomain.PF_BLOCK_IN_ROOM,
                new String[]{CleanupDomain.CLASS_BLOCK, CleanupDomain.CLASS_ROOM}, false);


        GroundedProp gp =  new GroundedProp(pf, new String[]{"block0", "room1"});
        symbolMap.put("R", gp);

        GLTLCompiler compiler = new GLTLCompiler(formula, symbolMap, envDomain);
        Domain compiledDomain = compiler.generateDomain();
        RewardFunction rf = compiler.generateRewardFunction();
        TerminalFunction tf = compiler.generateTerminalFunction();

        State initialCompiledState = compiler.addInitialTaskStateToEnvironmentState((OOState) s);
        // System.out.println(initialCompiledState.getCompleteStateDescription());

        HashableStateFactory hashingFactory = new SimpleHashableStateFactory();

        //begin planning in our compiled domain
        Planner planner = new ValueIteration((SADomain) compiledDomain, 1.0, hashingFactory, 0.0000001, 20000);
        Policy p = planner.planFromState(initialCompiledState);

        Episode ea = PolicyUtils.rollout(p, initialCompiledState, ((SADomain) compiledDomain).getModel());

        String yamlOut = ea.serialize();
        System.out.println(yamlOut);
        System.out.println("\n\n");
        Yaml yaml = new Yaml();
        Episode read = (Episode) yaml.load(yamlOut);
        System.out.println(read.actionString());
        System.out.println(read.state(0).toString());
        System.out.println(read.actionSequence.size());
        System.out.println(read.stateSequence.size());
    }
}
