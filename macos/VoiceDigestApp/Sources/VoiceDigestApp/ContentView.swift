import SwiftUI
import UniformTypeIdentifiers

enum MainTab: String, CaseIterable, Identifiable {
    case file = "Transcribe File"
    case live = "Live Call"
    var id: String { rawValue }
}

struct ContentView: View {
    @StateObject private var vm = TranscriptionViewModel()
    @ObservedObject var liveVM: LiveCaptureViewModel
    @AppStorage("hideDockIcon") private var hideDockIcon: Bool = false
    @State private var selectedTab: MainTab = .file
    @State private var isDropTargeted = false
    @State private var showingInfo = false

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            headerSection

            Picker("", selection: $selectedTab) {
                ForEach(MainTab.allCases) { tab in
                    Text(tab.rawValue).tag(tab)
                }
            }
            .pickerStyle(.segmented)
            .labelsHidden()

            if selectedTab == .file {
                dropZoneSection
                optionsSection
                outputFolderSection
                Spacer(minLength: 4)
                progressAndActionsSection
            } else {
                liveCallTabSection
            }
        }
        .padding(22)
        .sheet(isPresented: $showingInfo) {
            InfoView()
        }
    }

    private var headerSection: some View {
        HStack {
            VStack(alignment: .leading, spacing: 2) {
                HStack(spacing: 6) {
                    Text("VoiceDigest")
                        .font(.title2.bold())
                    Button {
                        showingInfo = true
                    } label: {
                        Image(systemName: "info.circle")
                            .font(.title3)
                            .foregroundStyle(.secondary)
                    }
                    .buttonStyle(.plain)
                    .help("View detailed info about settings, models, and RAM usage")
                }
                Text("100% on-device transcription, speaker diarization & acoustic cues")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            Spacer()
        }
    }

    private var dropZoneSection: some View {
        VStack(spacing: 8) {
            Image(systemName: vm.inputFileURL == nil ? "waveform.badge.plus" : "waveform.circle.fill")
                .font(.system(size: 32))
                .foregroundStyle(vm.inputFileURL == nil ? Color.accentColor : Color.green)

            Text(vm.inputFileURL == nil ? "Drop an audio or video file here" : vm.inputFileSummary)
                .font(.headline)
                .lineLimit(1)
                .truncationMode(.middle)

            if vm.inputFileURL == nil {
                Text("Supports .m4a, .mp3, .wav, .mp4 & more — tap ⓘ above for full list")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            Button(vm.inputFileURL == nil ? "Choose Audio / Video File…" : "Change File…") {
                vm.chooseInputFile()
            }
            .buttonStyle(.bordered)
            .disabled(vm.isRunning)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 22)
        .padding(.horizontal, 16)
        .background(
            RoundedRectangle(cornerRadius: 12)
                .fill(isDropTargeted ? Color.accentColor.opacity(0.12) : Color(nsColor: .controlBackgroundColor))
        )
        .overlay(
            RoundedRectangle(cornerRadius: 12)
                .strokeBorder(
                    isDropTargeted ? Color.accentColor : Color.secondary.opacity(0.3),
                    style: StrokeStyle(lineWidth: isDropTargeted ? 2 : 1.5, dash: [6])
                )
        )
        .onDrop(of: [.fileURL], isTargeted: $isDropTargeted) { providers in
            guard !vm.isRunning, let provider = providers.first else { return false }
            provider.loadItem(forTypeIdentifier: UTType.fileURL.identifier, options: nil) { item, _ in
                guard let data = item as? Data,
                      let url = URL(dataRepresentation: data, relativeTo: nil) else { return }
                Task { @MainActor in
                    vm.inputFileURL = url
                    vm.errorMessage = nil
                    vm.completedOutputURL = nil
                }
            }
            return true
        }
    }

    private var optionsSection: some View {
        GroupBox("Settings") {
            VStack(alignment: .leading, spacing: 12) {
                HStack {
                    Text("Model Size")
                        .frame(width: 110, alignment: .leading)
                    Picker("", selection: $vm.modelSize) {
                        ForEach(WhisperModelSize.allCases) { size in
                            Text(size.label).tag(size)
                        }
                    }
                    .labelsHidden()
                    .disabled(vm.isRunning)
                }

                HStack {
                    Text("Language")
                        .frame(width: 110, alignment: .leading)
                    Picker("", selection: $vm.language) {
                        ForEach(LanguageOption.allCases) { lang in
                            Text(lang.label).tag(lang)
                        }
                    }
                    .labelsHidden()
                    .frame(maxWidth: 180)
                    .disabled(vm.isRunning)

                    Spacer()

                    Stepper("Speakers: \(vm.speakers)", value: $vm.speakers, in: 1...10)
                        .disabled(vm.isRunning)
                }

                Divider()

                Toggle("Loudness Normalization (recommended for quiet calls)", isOn: $vm.normalizeLoudness)
                    .disabled(vm.isRunning)

                if vm.normalizeLoudness {
                    HStack(spacing: 10) {
                        Text("Target Loudness:")
                            .font(.subheadline)
                            .foregroundStyle(.secondary)
                        Slider(value: $vm.loudnessTargetLUFS, in: -24...(-10), step: 1)
                            .frame(maxWidth: 180)
                            .disabled(vm.isRunning)
                        Text("\(Int(vm.loudnessTargetLUFS.rounded())) LUFS")
                            .font(.subheadline.monospacedDigit())
                            .frame(width: 68, alignment: .trailing)
                    }
                    .padding(.leading, 20)
                }

                Toggle("AI Speaker Repair (Local Qwen 7B — fixes speaker swaps)", isOn: $vm.enableLLMRepair)
                    .disabled(vm.isRunning || vm.speakers < 2)

                Toggle("Publish AI Summary (Local Qwen 7B — prepends summary to .txt)", isOn: $vm.publishSummary)
                    .disabled(vm.isRunning)

                if vm.publishSummary {
                    VStack(alignment: .leading, spacing: 6) {
                        HStack {
                            Text("What should the summary focus on? (optional):")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                            Spacer()
                            Text("\(vm.summaryFocus.count)/160")
                                .font(.caption.monospacedDigit())
                                .foregroundStyle(vm.summaryFocus.count >= 160 ? .orange : .secondary)
                        }
                        TextField("e.g., Rate the interview, extract TODOs, or summarize podcast topics", text: $vm.summaryFocus)
                            .textFieldStyle(.roundedBorder)
                            .disabled(vm.isRunning)
                            .onChange(of: vm.summaryFocus) { _, newValue in
                                if newValue.count > 160 {
                                    vm.summaryFocus = String(newValue.prefix(160))
                                }
                            }
                        HStack(spacing: 6) {
                            summaryPresetChip("Interview Feedback", "Rate the candidate's interview performance, strengths, gaps, and suggested improvements.")
                            summaryPresetChip("TODOs", "Extract all action items, owners, decisions, and follow-up TODOs.")
                            summaryPresetChip("Podcast", "Summarize the main topics, key insights, and memorable takeaways from this podcast.")
                            summaryPresetChip("Meeting", "Summarize key discussion points, decisions made, and next steps.")
                        }
                    }
                    .padding(.leading, 20)
                }

                Toggle("Include Acoustic & Prosody Analysis (pace, pauses, stress)", isOn: $vm.enableProsody)
                    .disabled(vm.isRunning)

                if vm.enableProsody {
                    Toggle("Fast DSP-only Prosody (skip heavy neural SER model)", isOn: $vm.fastDspOnlyProsody)
                        .padding(.leading, 20)
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                        .disabled(vm.isRunning)
                }
            }
            .padding(8)
        }
    }

    private func summaryPresetChip(_ title: String, _ prompt: String) -> some View {
        Button(title) {
            vm.summaryFocus = String(prompt.prefix(160))
        }
        .buttonStyle(.bordered)
        .controlSize(.small)
        .disabled(vm.isRunning)
    }

    private var outputFolderSection: some View {
        HStack {
            Text("Output Folder:")
                .font(.subheadline.weight(.medium))
            Text(vm.outputFolderURL.path)
                .font(.subheadline)
                .foregroundStyle(.secondary)
                .lineLimit(1)
                .truncationMode(.head)
            Spacer()
            Button("Choose…") {
                vm.chooseOutputFolder()
            }
            .disabled(vm.isRunning)
        }
        .padding(.horizontal, 4)
    }

    private var progressAndActionsSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            VStack(alignment: .leading, spacing: 6) {
                ProgressView(value: vm.progress, total: 100.0)
                    .progressViewStyle(.linear)

                HStack {
                    Text(vm.stageLabel)
                        .font(.subheadline.weight(.medium))
                    if !vm.detailLabel.isEmpty {
                        Text("— \(vm.detailLabel)")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                            .lineLimit(1)
                            .truncationMode(.tail)
                    }
                    Spacer()
                    if vm.isRunning || vm.elapsedSeconds > 0 {
                        Text(vm.formattedElapsed)
                            .font(.caption.monospacedDigit())
                            .foregroundStyle(.secondary)
                    }
                }
            }

            if let err = vm.errorMessage {
                Text(err)
                    .font(.caption)
                    .foregroundStyle(.red)
                    .lineLimit(3)
            }

            HStack {
                if vm.completedOutputURL != nil {
                    Button("Open Transcript") {
                        vm.openCompletedTranscript()
                    }
                    Button("Reveal in Finder") {
                        vm.revealInFinder()
                    }
                }

                Spacer()

                if vm.isRunning {
                    Button("Cancel", role: .destructive) {
                        vm.cancel()
                    }
                } else {
                    Button("Start Transcription") {
                        vm.start()
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(vm.inputFileURL == nil)
                    .keyboardShortcut(.defaultAction)
                }
            }
        }
    }

    private var liveCallTabSection: some View {
        VStack(alignment: .leading, spacing: 16) {
            // Live capture hero box
            HStack(spacing: 14) {
                Image(systemName: liveVM.isRecording ? "waveform.circle.fill" : "mic.circle.fill")
                    .font(.system(size: 36))
                    .foregroundStyle(liveVM.isRecording ? Color.green : Color.accentColor)

                VStack(alignment: .leading, spacing: 2) {
                    Text(liveVM.isRecording ? "Capturing Call Audio — \(liveVM.formattedElapsed)" : "Dual-Channel Call Capture")
                        .font(.headline)
                    Text("Captures your mic (Me) and system audio (Other) locally. You can also Start/Stop anytime from the top menu bar icon.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
                Spacer()
            }
            .padding(16)
            .background(Color(nsColor: .controlBackgroundColor), in: RoundedRectangle(cornerRadius: 12))

            if let warn = liveVM.systemAudioWarning {
                Text(warn)
                    .font(.caption)
                    .foregroundStyle(.orange)
            }
            if let err = liveVM.errorMessage {
                Text(err)
                    .font(.caption)
                    .foregroundStyle(.red)
            }

            GroupBox("Live Call Settings") {
                VStack(alignment: .leading, spacing: 11) {
                    HStack {
                        Text("Processing Mode")
                            .frame(width: 120, alignment: .leading)
                        Picker("", selection: $liveVM.mode) {
                            ForEach(CaptureProcessingMode.allCases) { m in
                                Text(m.rawValue).tag(m)
                            }
                        }
                        .pickerStyle(.segmented)
                        .labelsHidden()
                        .disabled(liveVM.isRecording)
                    }

                    HStack {
                        Text("Whisper Model")
                            .frame(width: 120, alignment: .leading)
                        if liveVM.mode == .liveRolling60s {
                            Picker("", selection: $liveVM.liveChunkModelSize) {
                                Text("Base (Fastest, ~150 MB)").tag(WhisperModelSize.base)
                                Text("Small (Balanced, ~500 MB)").tag(WhisperModelSize.small)
                            }
                            .labelsHidden()
                            .disabled(liveVM.isRecording)
                        } else {
                            Picker("", selection: $liveVM.modelSize) {
                                ForEach(WhisperModelSize.allCases) { size in
                                    Text(size.label).tag(size)
                                }
                            }
                            .labelsHidden()
                            .disabled(liveVM.isRecording)
                        }
                    }

                    HStack {
                        Text("Other Side")
                            .frame(width: 120, alignment: .leading)
                        Stepper(
                            liveVM.remoteSpeakers == 1 ? "1 person (1:1 exact)" : "\(liveVM.remoteSpeakers) people (1:N diarized)",
                            value: $liveVM.remoteSpeakers,
                            in: 1...8
                        )
                        .disabled(liveVM.isRecording)
                        Spacer()
                    }

                    Divider()

                    Toggle("Post-call Acoustic & Prosody Analysis (DSP)", isOn: $liveVM.enableProsody)
                        .disabled(liveVM.isRecording)

                    Toggle("Post-call AI Summary (Local Qwen 7B)", isOn: $liveVM.publishSummary)
                        .disabled(liveVM.isRecording)

                    if liveVM.publishSummary {
                        TextField("Summary focus (optional, max 160 chars)", text: $liveVM.summaryFocus)
                            .textFieldStyle(.roundedBorder)
                            .padding(.leading, 20)
                            .disabled(liveVM.isRecording)
                            .onChange(of: liveVM.summaryFocus) { _, newValue in
                                if newValue.count > 160 {
                                    liveVM.summaryFocus = String(newValue.prefix(160))
                                }
                            }
                    }

                    Divider()

                    Toggle("Menu Bar Only (Hide Dock Icon)", isOn: $hideDockIcon)
                        .onChange(of: hideDockIcon) { _, hidden in
                            NSApplication.shared.setActivationPolicy(hidden ? .accessory : .regular)
                        }
                }
                .padding(8)
            }

            HStack {
                Text("Output Folder:")
                    .font(.subheadline.weight(.medium))
                Text(liveVM.outputFolderURL.path)
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
                    .truncationMode(.head)
                Spacer()
                Button("Choose…") { liveVM.chooseOutputFolder() }
                    .disabled(liveVM.isRecording)
            }
            .padding(.horizontal, 4)

            HStack {
                Text("Temporary Audio: \(liveVM.formattedTempUsage)")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Spacer()
                Button("Clear Temp Files") { liveVM.clearTempFiles() }
                    .buttonStyle(.bordered)
                    .controlSize(.small)
            }
            .padding(.horizontal, 4)

            if !liveVM.jobs.isEmpty {
                GroupBox("Recent Captures") {
                    VStack(alignment: .leading, spacing: 8) {
                        ForEach(liveVM.jobs.prefix(3)) { job in
                            VStack(alignment: .leading, spacing: 4) {
                                HStack {
                                    Text(job.title)
                                        .font(.subheadline)
                                        .lineLimit(1)
                                    Spacer()
                                    if let url = job.outputURL, job.isFinished {
                                        Button("Open") { liveVM.openOutput(url) }
                                        Button("Finder") { liveVM.revealOutput(url) }
                                    } else if let err = job.error {
                                        Text(err).font(.caption).foregroundStyle(.red).lineLimit(1)
                                    } else {
                                        Text("\(Int(job.percent))% — \(job.stage)")
                                            .font(.caption.monospacedDigit())
                                            .foregroundStyle(.secondary)
                                    }
                                }
                                if !job.isFinished {
                                    ProgressView(value: job.percent, total: 100.0)
                                        .progressViewStyle(.linear)
                                }
                            }
                        }
                    }
                    .padding(6)
                }
            }

            Spacer(minLength: 4)

            HStack {
                Spacer()
                if liveVM.isRecording {
                    Button("Cancel", role: .destructive) {
                        liveVM.cancelCallCapture()
                    }
                    Button("Stop & Transcribe") {
                        liveVM.stopCallCapture()
                    }
                    .buttonStyle(.borderedProminent)
                } else {
                    Button("Start Call Capture") {
                        liveVM.startCallCapture()
                    }
                    .buttonStyle(.borderedProminent)
                }
            }
        }
    }
}
