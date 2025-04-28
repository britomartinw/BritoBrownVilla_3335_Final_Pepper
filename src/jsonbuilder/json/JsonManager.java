package jsonbuilder.json;

import java.io.*;
import java.util.*;
import java.lang.reflect.Type;

import com.google.gson.*;
import com.google.gson.reflect.TypeToken;

import jsonbuilder.database.Inntent;

public class JsonManager {
    private Gson gson;
    private String filePath;

    public JsonManager() {
        this.gson = new GsonBuilder().setPrettyPrinting().create();
        this.filePath = "json/intents.json";
    }

    public boolean addQuestion(String question, String answer, ArrayList<String> patterns) {
        try {
            // Read existing JSON file
            ArrayList<Inntent> existinQuestions = new ArrayList<>();
            FileReader reader = new FileReader(this.filePath);
            Type listType = new TypeToken<List<Inntent>>() {
            }.getType();
            existinQuestions = gson.fromJson(reader, listType);
            reader.close();

            if (existinQuestions == null) {
                existinQuestions = new ArrayList<>(); // handle empty file
            }

            // Add a new person
            existinQuestions.add(new Inntent(question, answer, patterns));

            // Write updated list back to the file
            try (FileWriter writer = new FileWriter(this.filePath)) {
                gson.toJson(existinQuestions, writer);
            }

        } catch (IOException e) {
            System.out.println("File not found");
            try {
                File file = new File(this.filePath);
                file.createNewFile();
                System.out.println("Creating File: " + file.getName());
            } catch (IOException e2) {
                System.out.println("An error occurred.");
                e2.printStackTrace();
            }
            addQuestion(question, answer, patterns);
        }
        return true;
    }

}
