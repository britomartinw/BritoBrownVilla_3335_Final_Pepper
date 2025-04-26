package jsonbuilder.database;

import java.io.File;
import java.io.IOException;
import java.sql.SQLException;
import java.util.ArrayList;

import jsonbuilder.json.JsonManager;

public class BusinessLayer {
    DataMgr dataMgr;
    JsonManager jsonManager;
    private String filePath = "json/intents.json";

    public BusinessLayer(String username, String password) {
        this.dataMgr = new DataMgr(username, password);
        this.jsonManager = new JsonManager();
        try {
            dataMgr.getConnection();
        } catch (SQLException e) {
            e.printStackTrace();
        }
    }

    public String getQuestion(int myQuestionId) {
        return this.dataMgr.getQuestion(myQuestionId);
    }

    public String getAnswer(int myQuestionId) {
        return this.dataMgr.getAnswer(myQuestionId);
    }

    public ArrayList<String> getPatterns(int myQuestionId) {
        return this.dataMgr.getPatterns(myQuestionId);
    }

    public int getNumberOfQuestions() {
        return this.dataMgr.getNumberOfQuestions();
    }

    public void addAllQuestion() {
        // Deletes all the intents to assure no duplicates when adding all the questions
        this.deleteAllIntents();
        int getNumberOfQuestions = this.getNumberOfQuestions();
        String question;
        String answer;
        ArrayList<String> patterns;
        for (int i = 1; i <= getNumberOfQuestions; i++) {
            question = this.getQuestion(i);
            answer = this.getAnswer(i);
            patterns = this.getPatterns(i);
            this.jsonManager.addQuestion(question, answer, patterns);
        }

    }

    private void deleteAllIntents() {
        try {
            File myObj = new File(this.filePath);
            myObj.createNewFile();
            System.out.println("Deleting File: " + myObj.getName());
        } catch (IOException e2) {
            System.out.println("An error occurred.");
            e2.printStackTrace();
        }
    }

}
