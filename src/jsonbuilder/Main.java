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
        
        bLayer.addAllQuestion();
    }
}