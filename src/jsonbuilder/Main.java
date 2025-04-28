package jsonbuilder;
import java.util.Scanner;

import jsonbuilder.database.BusinessLayer;

public class Main {
    public static void main(String[] args) {

        Scanner scanner = new Scanner(System.in);
        System.out.print("Enter username: ");
        String username = scanner.next();
        System.out.print("Enter password: ");
        String password = scanner.next();
        BusinessLayer bLayer = new BusinessLayer(username, password);
        scanner.close();
        System.out.println(bLayer.getTag(5));
        // System.out.println(bLayer.getAnswer(1));
        System.out.println(bLayer.getKeywords(5));
        System.out.println(bLayer.getNumberOfIntents());
        bLayer.addAllIntents();
    }
}