package jsonbuilder.database;

import java.util.ArrayList;

public class Inntent {
    private String tag;
    private String answer;
    private ArrayList<String> keywords;

    public Inntent(String question, String answer, ArrayList<String> patterns) {
        this.tag = question;
        this.answer = answer;
        this.keywords = patterns;
    }

    public String getTag() {
        return tag;
    }

    public String getAnswer() {
        return answer;
    }

    public ArrayList<String> getKeywords() {
        return keywords;
    }

}
