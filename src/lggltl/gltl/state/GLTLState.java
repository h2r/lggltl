package lggltl.gltl.state;

import burlap.mdp.core.oo.state.MutableOOState;
import burlap.mdp.core.oo.state.OOState;
import burlap.mdp.core.oo.state.ObjectInstance;
import burlap.mdp.core.state.MutableState;
import burlap.mdp.core.state.State;
import lggltl.gltl.GLTLCompiler;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by dilip and nakul on 10/13/17.
 * This is a wrapper state around the base state that holds on to the
 * GLTL spec attribute along with the environment state.
 */
public class GLTLState implements MutableOOState {

    public int spec;
    public String StateType = "type";

    public OOState envState;

    public GLTLState(OOState envState){
        this.envState = envState;
        this.spec = 2;
    }

    public GLTLState(OOState envState, int spec){
        this.envState = envState;
        this.spec = spec;
    }

    public GLTLState(){}

    public State getBaseState(){
        return envState;
    }

    @Override
    public MutableOOState addObject(ObjectInstance objectInstance) {
        throw new RuntimeException("Cannot add objects to GLTLState.");
    }

    @Override
    public MutableOOState removeObject(String s) {
        throw new RuntimeException("Cannot remove objects from GLTLState.");
    }

    @Override
    public MutableOOState renameObject(String s, String s1) {
        throw new RuntimeException("Cannot rename objects from GLTLState.");
    }

    @Override
    public int numObjects() {
        //TODO: one object is the spec and another is the base state?? Do not think we will need this method!
        return 2;
    }

    @Override
    public ObjectInstance object(String s) {
        throw new RuntimeException("Cannot fetch objects from GLTLState.");
    }

    @Override
    public List<ObjectInstance> objects() {
        List<ObjectInstance> objects = new ArrayList<>();
        objects = this.envState.objects();
        objects.add(new PseudoObject(GLTLCompiler.CLASSSPEC, this.spec));
        return objects;
    }

    @Override
    public List<ObjectInstance> objectsOfClass(String s) {
        throw new RuntimeException("Cannot fetch objects from GLTLState.");
    }

    @Override
    public MutableState set(Object o, Object o1) {
        throw new RuntimeException("Cannot set objects in GLTLState.");
    }

    @Override
    public List<Object> variableKeys() {
        throw new RuntimeException("Cannot fetch objects from GLTLState.");
    }

    @Override
    public Object get(Object o) {
        throw new RuntimeException("Cannot fetch objects from GLTLState.");
    }

    @Override
    public GLTLState copy() {
        return new GLTLState((OOState)envState.copy(), Integer.valueOf(spec));
    }
}
