package lggltl.gltl.state;

import burlap.mdp.core.oo.state.MutableOOState;
import burlap.mdp.core.oo.state.ObjectInstance;
import burlap.mdp.core.state.MutableState;
import burlap.mdp.core.state.State;

import java.util.List;

/**
 * Created by dilip on 10/13/17.
 */
public class GLTLSpec implements MutableOOState {

    public int spec;

    public GLTLSpec(int spec){
        this.spec = spec;
    }


    @Override
    public MutableOOState addObject(ObjectInstance objectInstance) {
        return null;
    }

    @Override
    public MutableOOState removeObject(String s) {
        return null;
    }

    @Override
    public MutableOOState renameObject(String s, String s1) {
        return null;
    }

    @Override
    public int numObjects() {
        return 0;
    }

    @Override
    public ObjectInstance object(String s) {
        return null;
    }

    @Override
    public List<ObjectInstance> objects() {
        return null;
    }

    @Override
    public List<ObjectInstance> objectsOfClass(String s) {
        return null;
    }

    @Override
    public MutableState set(Object o, Object o1) {
        return null;
    }

    @Override
    public List<Object> variableKeys() {
        return null;
    }

    @Override
    public Object get(Object o) {
        return null;
    }

    @Override
    public State copy() {
        return null;
    }
}
