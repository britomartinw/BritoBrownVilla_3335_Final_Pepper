package jsonbuilder.database;

import java.io.File;
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

    public String getTag(int myIntentId) {
        return this.dataMgr.getTag(myIntentId);
    }

    public String getAnswer(int myIntentId) {
        return this.dataMgr.getAnswer(myIntentId);
    }

    public ArrayList<String> getKeywords(int myIntentId) {
        return this.dataMgr.getKeywords(myIntentId);
    }

    public int getNumberOfIntents() {
        return this.dataMgr.getNumberOfIntents();
    }

    public void addAllIntents() {
        // Deletes all the intents to assure no duplicates when adding all the questions
        this.deleteAllIntents();
        int getNumberOfQuestions = this.getNumberOfIntents();
        String question;
        String answer;
        ArrayList<String> patterns;
        for (int i = 1; i <= getNumberOfQuestions; i++) {
            question = this.getTag(i);
            answer = this.getAnswer(i);
            patterns = this.getKeywords(i);
            this.jsonManager.addQuestion(question, answer, patterns);
        }

    }

    private void deleteAllIntents() {
        File file = new File(this.filePath);
        if (file.exists()) {
            file.delete();
            System.out.println("Deleting File: " + file.getName());
        }
    }

}
