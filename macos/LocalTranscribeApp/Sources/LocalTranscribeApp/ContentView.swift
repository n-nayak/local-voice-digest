import SwiftUI
import UniformTypeIdentifiers

struct ContentView: View {
    @StateObject private var vm = TranscriptionViewModel()
    @State private var isDropTargeted = false
    @State private var showingInfo = false

    var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            headerSection
            dropZoneSection
            optionsSection
            outputFolderSection
            Spacer(minLength: 4)
            progressAndActionsSection
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
                    Text("Local Audio Transcribe")
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
            Label("Offline", systemImage: "lock.shield.fill")
                .font(.caption.weight(.medium))
                .padding(.horizontal, 10)
                .padding(.vertical, 4)
                .background(.green.opacity(0.14), in: Capsule())
                .foregroundStyle(.green)
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
                            .onChange(of: vm.summaryFocus) { newValue in
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
}
