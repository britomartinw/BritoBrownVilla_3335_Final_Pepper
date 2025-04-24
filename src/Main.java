import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.Scanner;

public class Main {
    public static void main(String[] args) {

        Scanner scanner = new Scanner(System.in);
        System.out.print("Enter username: ");
        String username = scanner.next();
        System.out.print("Enter password: ");
        String password = scanner.next();
        BusinessLayer bLayer = new BusinessLayer(username, password);
        // System.out.println("Question: " + bLayer.getQuestion(1));
        // System.out.println("Answer: " + bLayer.getAnswer(1));
        // System.out.println("Patterns: ");
        // ArrayList<String> patterns = bLayer.getPatterns(1);
        // for (int i = 0; i < patterns.size(); i++) {
        //     System.out.println("\t" + patterns.get(i));
        // }
        System.out.println("Number of Questions: " + bLayer.getNumberOfQuestions());
        scanner.close();
    }
}