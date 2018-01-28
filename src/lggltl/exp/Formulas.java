package lggltl.exp;

/**
 * Created by dilip on 10/31/17.
 */
public class Formulas {

    protected static final String EVENT_RED = "F4R";
    protected static final String EVENT_GREEN = "F4C";
    protected static final String EVENT_BLUE = "F4B";
    protected static final String EVENT_YELLOW = "F4Y";

    protected static final String ALW_EVENT_GREEN_AND_EVENT_BLUE = "G4F4&CF4B"; //Always eventually (Green and eventually Blue)
    protected static final String ALW_EVENT_RED_AND_EVENT_BLUE = "G4F4&RF4B"; //Always eventually (Red and eventually Blue)
    protected static final String ALW_EVENT_BLUE_AND_EVENT_YELLOW = "G4F4&BF4Y"; //Always eventually (Blue and eventually Yellow)

    protected static final String EVENT_BLOCK2GREEN = "F4X";
    protected static final String EVENT_BLOCK2GREEN_AND_NEVER_BLUE = "&F4XG4!B";
    protected static final String EVENT_BLOCK2GREEN_AND_NEVER_BLUE_UNTIL_BLOCK2GREEN = "&F4XU4G4!BX";
    protected static final String EVENT_BLOCK2GREEN_AND_NEVER_YELLOW = "&F4XG4!Y";

    protected static final String ROTATE_FOUR_ROOMS = "G4F4&BF4&CF4&YF4R";
}
