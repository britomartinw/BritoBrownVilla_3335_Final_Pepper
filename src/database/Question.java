package database;

import java.util.ArrayList;

public class Question {
    private String question;
    private String answer;
    private ArrayList<String> patterns;

    public Question(String question, String answer, ArrayList<String> patterns) {
        this.question = question;
        this.answer = answer;
        this.patterns = patterns;
    }

    public String getQuestion() {
        return question;
    }

    public String getAnswer() {
        return answer;
    }

    public ArrayList<String> getPatterns() {
        return patterns;
    }

}
