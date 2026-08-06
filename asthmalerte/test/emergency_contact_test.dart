import 'package:asthmalerte/models/emergency_contact.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  EmergencyContact contact(String phone) =>
      EmergencyContact(id: '1', name: 'Alex', phone: phone);

  group('dialablePhone', () {
    test('supprime espaces, points et tirets', () {
      expect(contact('06 12.34-56 78').dialablePhone, '0612345678');
    });

    test('conserve un + en tête', () {
      expect(contact('+33 6 12 34 56 78').dialablePhone, '+33612345678');
    });

    test('ne garde pas les + internes', () {
      expect(contact('06+12+34').dialablePhone, '061234');
    });

    test('gère les parenthèses', () {
      expect(contact('(33) 612-345-678').dialablePhone, '33612345678');
    });
  });

  group('isValid', () {
    test('refuse un numéro trop court', () {
      expect(contact('123').isValid, isFalse);
    });

    test('refuse un nom vide', () {
      const c = EmergencyContact(id: '1', name: '  ', phone: '0612345678');
      expect(c.isValid, isFalse);
    });

    test('accepte un contact complet', () {
      expect(contact('0612345678').isValid, isTrue);
    });
  });

  test('sérialisation aller-retour', () {
    const original = EmergencyContact(
      id: 'abc',
      name: 'Marie',
      phone: '+33612345678',
      relation: 'Sœur',
      sendSms: true,
      callable: false,
    );
    final restored = EmergencyContact.fromJson(original.toJson());

    expect(restored.id, original.id);
    expect(restored.name, original.name);
    expect(restored.phone, original.phone);
    expect(restored.relation, original.relation);
    expect(restored.callable, isFalse);
  });
}
