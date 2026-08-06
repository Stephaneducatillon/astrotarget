import '../models/alert_event.dart';
import '../models/emergency_contact.dart';
import '../state/app_state.dart';
import 'alert_message.dart';
import 'location_service.dart';
import 'messaging_service.dart';

enum AlertStep { locating, composing, sending, calling, done }

/// Résultat d'une alerte, affiché à l'écran après l'envoi.
class AlertOutcome {
  AlertOutcome({
    required this.message,
    required this.recipients,
    this.position,
    this.locationError,
    this.smsOpened = false,
    this.callPlaced = false,
  });

  final String message;
  final List<EmergencyContact> recipients;
  final ResolvedPosition? position;
  final String? locationError;
  final bool smsOpened;
  final bool callPlaced;
}

/// Orchestration du déclenchement : position → message → SMS → appel →
/// historique. Chaque étape est signalée via [onStep] pour l'affichage.
class AlertService {
  AlertService({
    required this.appState,
    LocationService? location,
    MessagingService? messaging,
  })  : _location = location ?? LocationService(),
        _messaging = messaging ?? MessagingService();

  final AppState appState;
  final LocationService _location;
  final MessagingService _messaging;

  Future<AlertOutcome> trigger({void Function(AlertStep)? onStep}) async {
    final settings = appState.settings;
    final recipients = appState.smsContacts;

    ResolvedPosition? position;
    String? locationError;

    if (settings.shareLocation) {
      onStep?.call(AlertStep.locating);
      try {
        position = await _location.resolve();
      } on LocationFailure catch (e) {
        locationError = e.message;
      } catch (e) {
        locationError = 'Position indisponible ($e).';
      }
    }

    onStep?.call(AlertStep.composing);
    final now = DateTime.now();
    final message = buildAlertMessage(
      settings: settings,
      profile: appState.profile,
      now: now,
      position: position,
    );

    onStep?.call(AlertStep.sending);
    var smsOpened = false;
    if (recipients.isNotEmpty) {
      final numbers = recipients.map((c) => c.dialablePhone).toList();
      if (settings.oneSmsPerContact) {
        smsOpened = await _messaging.sendSmsIndividually(numbers, message) > 0;
      } else {
        smsOpened = await _messaging.sendSms(numbers, message);
      }
    }

    var callPlaced = false;
    final callTarget = appState.primaryCallContact;
    if (settings.autoCallAfterSms && callTarget != null) {
      onStep?.call(AlertStep.calling);
      callPlaced = await _messaging.call(callTarget.dialablePhone);
    }

    onStep?.call(AlertStep.done);

    await appState.addAlert(AlertEvent(
      id: now.microsecondsSinceEpoch.toString(),
      date: now,
      recipients: recipients.map((c) => c.name).toList(),
      latitude: position?.latitude,
      longitude: position?.longitude,
      address: position?.address ?? '',
      message: message,
      called: callPlaced,
    ));

    return AlertOutcome(
      message: message,
      recipients: recipients,
      position: position,
      locationError: locationError,
      smsOpened: smsOpened,
      callPlaced: callPlaced,
    );
  }

  /// « Je vais mieux » : rassure les mêmes proches.
  Future<bool> sendAllClear() async {
    final recipients = appState.smsContacts;
    if (recipients.isEmpty) return false;
    final message = buildAllClearMessage(
      profile: appState.profile,
      now: DateTime.now(),
    );
    return _messaging.sendSms(
      recipients.map((c) => c.dialablePhone).toList(),
      message,
    );
  }

  Future<bool> call(EmergencyContact contact) =>
      _messaging.call(contact.dialablePhone);

  /// Appel des secours (15 / 112 en France).
  Future<bool> callEmergencyServices({String number = '112'}) =>
      _messaging.call(number);

  Future<bool> openMaps(double latitude, double longitude) =>
      _messaging.openMaps(latitude, longitude);
}
