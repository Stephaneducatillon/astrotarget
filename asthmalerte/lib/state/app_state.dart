import 'package:flutter/foundation.dart';

import '../models/alert_event.dart';
import '../models/app_settings.dart';
import '../models/emergency_contact.dart';
import '../models/medical_profile.dart';
import '../services/storage_service.dart';

/// État global : contacts, fiche médicale, réglages, historique.
class AppState extends ChangeNotifier {
  AppState({StorageService? storage})
      : _storage = storage ?? StorageService();

  final StorageService _storage;

  List<EmergencyContact> _contacts = [];
  MedicalProfile _profile = const MedicalProfile();
  AppSettings _settings = const AppSettings();
  List<AlertEvent> _history = [];
  bool _onboarded = false;
  bool _loaded = false;

  List<EmergencyContact> get contacts => List.unmodifiable(_contacts);
  MedicalProfile get profile => _profile;
  AppSettings get settings => _settings;
  List<AlertEvent> get history => List.unmodifiable(_history);
  bool get onboarded => _onboarded;
  bool get isReady => _loaded;

  /// Prêt à alerter : au moins un proche joignable par SMS.
  bool get canAlert => smsContacts.isNotEmpty;

  List<EmergencyContact> get smsContacts =>
      _contacts.where((c) => c.sendSms && c.isValid).toList();

  /// Contact appelé en priorité : le premier de la liste qui est appelable.
  EmergencyContact? get primaryCallContact {
    for (final c in _contacts) {
      if (c.callable && c.isValid) return c;
    }
    return null;
  }

  Future<void> load() async {
    _contacts = await _storage.loadContacts();
    _profile = await _storage.loadProfile();
    _settings = await _storage.loadSettings();
    _history = await _storage.loadHistory();
    _onboarded = await _storage.isOnboarded();
    _loaded = true;
    notifyListeners();
  }

  Future<void> upsertContact(EmergencyContact contact) async {
    final index = _contacts.indexWhere((c) => c.id == contact.id);
    if (index >= 0) {
      _contacts[index] = contact;
    } else {
      _contacts.add(contact);
    }
    await _storage.saveContacts(_contacts);
    notifyListeners();
  }

  Future<void> removeContact(String id) async {
    _contacts.removeWhere((c) => c.id == id);
    await _storage.saveContacts(_contacts);
    notifyListeners();
  }

  /// Réordonne la liste : le premier contact est celui qu'on appelle d'abord.
  Future<void> reorderContacts(int oldIndex, int newIndex) async {
    if (newIndex > oldIndex) newIndex -= 1;
    final contact = _contacts.removeAt(oldIndex);
    _contacts.insert(newIndex, contact);
    await _storage.saveContacts(_contacts);
    notifyListeners();
  }

  Future<void> saveProfile(MedicalProfile profile) async {
    _profile = profile;
    await _storage.saveProfile(profile);
    notifyListeners();
  }

  Future<void> saveSettings(AppSettings settings) async {
    _settings = settings;
    await _storage.saveSettings(settings);
    notifyListeners();
  }

  Future<void> addAlert(AlertEvent event) async {
    _history = [event, ..._history];
    await _storage.saveHistory(_history);
    notifyListeners();
  }

  Future<void> clearHistory() async {
    _history = [];
    await _storage.saveHistory(_history);
    notifyListeners();
  }

  Future<void> completeOnboarding() async {
    _onboarded = true;
    await _storage.setOnboarded(true);
    notifyListeners();
  }
}
