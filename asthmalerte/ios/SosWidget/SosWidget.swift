import SwiftUI
import WidgetKit

/// Widget « SOS » de l'écran d'accueil iOS.
///
/// Il n'exécute aucune logique : un appui ouvre l'application sur l'URI
/// `asthmalerte://sos`, que Flutter intercepte pour afficher l'écran
/// d'alerte (voir HomeEntryService côté Dart).
struct SosEntry: TimelineEntry {
    let date: Date
    let subtitle: String
}

struct SosProvider: TimelineProvider {
    /// Doit correspondre à l'App Group partagé avec l'application.
    static let appGroupId = "group.com.astrotarget.asthmalerte"

    func placeholder(in context: Context) -> SosEntry {
        SosEntry(date: Date(), subtitle: "Prévenir mes proches")
    }

    func getSnapshot(in context: Context, completion: @escaping (SosEntry) -> Void) {
        completion(entry())
    }

    func getTimeline(in context: Context, completion: @escaping (Timeline<SosEntry>) -> Void) {
        // Contenu quasi statique : on rafraîchit seulement quand l'app le demande.
        completion(Timeline(entries: [entry()], policy: .never))
    }

    private func entry() -> SosEntry {
        let defaults = UserDefaults(suiteName: Self.appGroupId)
        let subtitle = defaults?.string(forKey: "sos_subtitle") ?? "Prévenir mes proches"
        return SosEntry(date: Date(), subtitle: subtitle)
    }
}

struct SosWidgetEntryView: View {
    var entry: SosEntry

    var body: some View {
        VStack(spacing: 4) {
            Image(systemName: "exclamationmark.triangle.fill")
                .font(.system(size: 26, weight: .bold))
                .foregroundColor(.white)
            Text("SOS")
                .font(.system(size: 38, weight: .black))
                .foregroundColor(.white)
            Text(entry.subtitle)
                .font(.system(size: 12, weight: .medium))
                .foregroundColor(.white.opacity(0.9))
                .lineLimit(1)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(Color(red: 0.827, green: 0.125, blue: 0.161))
        .widgetURL(URL(string: "asthmalerte://sos"))
    }
}

@main
struct SosWidget: Widget {
    let kind = "SosWidget"

    var body: some WidgetConfiguration {
        StaticConfiguration(kind: kind, provider: SosProvider()) { entry in
            if #available(iOSApplicationExtension 17.0, *) {
                SosWidgetEntryView(entry: entry)
                    .containerBackground(for: .widget) {
                        Color(red: 0.827, green: 0.125, blue: 0.161)
                    }
            } else {
                SosWidgetEntryView(entry: entry)
            }
        }
        .configurationDisplayName("Alerte SOS")
        .description("Prévient vos proches et partage votre position en un appui.")
        .supportedFamilies([.systemSmall, .systemMedium])
    }
}
