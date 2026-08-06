import 'dart:convert';

import 'package:shared_preferences/shared_preferences.dart';

import '../models/alert_event.dart';
import '../models/app_settings.dart';
import '../models/emergency_contact.dart';
import '../models/medical_profile.dart';

/// Persistance locale (SharedPreferences). Rien ne sort du téléphone.
class StorageService {
  static const _kContacts = 'contacts';
  static const _kProfile = 'profile';
  static const _kSettings = 'settings';
  static const _kHistory = 'history';
  static const _kOnboarded = 'onboarded';

  static const int maxHistoryEntries = 50;

  SharedPreferences? _prefs;

  Future<SharedPreferences> get _p async =>
      _prefs ??= await SharedPreferences.getInstance();

  Future<List<EmergencyContact>> loadContacts() async {
    final raw = (await _p).getString(_kContacts);
    if (raw == null || raw.isEmpty) return [];
    final list = jsonDecode(raw) as List<dynamic>;
    return list
        .map((e) => EmergencyContact.fromJson(e as Map<String, dynamic>))
        .toList();
  }

  Future<void> saveContacts(List<EmergencyContact> contacts) async {
    await (await _p).setString(
      _kContacts,
      jsonEncode(contacts.map((c) => c.toJson()).toList()),
    );
  }

  Future<MedicalProfile> loadProfile() async {
    final raw = (await _p).getString(_kProfile);
    if (raw == null || raw.isEmpty) return const MedicalProfile();
    return MedicalProfile.fromJson(jsonDecode(raw) as Map<String, dynamic>);
  }

  Future<void> saveProfile(MedicalProfile profile) async {
    await (await _p).setString(_kProfile, jsonEncode(profile.toJson()));
  }

  Future<AppSettings> loadSettings() async {
    final raw = (await _p).getString(_kSettings);
    if (raw == null || raw.isEmpty) return const AppSettings();
    return AppSettings.fromJson(jsonDecode(raw) as Map<String, dynamic>);
  }

  Future<void> saveSettings(AppSettings settings) async {
    await (await _p).setString(_kSettings, jsonEncode(settings.toJson()));
  }

  Future<List<AlertEvent>> loadHistory() async {
    final raw = (await _p).getString(_kHistory);
    if (raw == null || raw.isEmpty) return [];
    final list = jsonDecode(raw) as List<dynamic>;
    return list
        .map((e) => AlertEvent.fromJson(e as Map<String, dynamic>))
        .toList();
  }

  Future<void> saveHistory(List<AlertEvent> history) async {
    final trimmed = history.take(maxHistoryEntries).toList();
    await (await _p).setString(
      _kHistory,
      jsonEncode(trimmed.map((e) => e.toJson()).toList()),
    );
  }

  Future<bool> isOnboarded() async => (await _p).getBool(_kOnboarded) ?? false;

  Future<void> setOnboarded(bool value) async {
    await (await _p).setBool(_kOnboarded, value);
  }
}
