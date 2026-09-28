"""Attach immutable source locations to answers at the runtime boundary."""
from urllib.parse import urlencode


class VersionedAnswerGenerator:
    def __init__(self, generator, scan, analysis, fingerprint):
        self.generator = generator
        self.root = scan.metadata.root_path
        self.fingerprint = fingerprint.value
        self.files = {f.file_id: f for f in scan.files}
        self.paths = {f.relative_path: f for f in scan.files}
        self.symbols = {s.symbol_id: s for s in analysis.symbols}

    def generate(self, result):
        answer = self.generator.generate(result)
        citations = []
        for citation in answer.citations:
            updates = {}
            path = citation.relative_path
            start, end = citation.start_line, citation.end_line
            if not path and citation.source_symbol_id in self.symbols:
                symbol = self.symbols[citation.source_symbol_id]
                file = self.files[symbol.file_id]
                path, start, end = file.relative_path, symbol.start_line, symbol.end_line
            file = self.paths.get(path)
            if file is not None and start is not None and end is not None:
                updates = dict(relative_path=path, start_line=start, end_line=end,
                               source_hash=file.content_hash, index_fingerprint=self.fingerprint,
                               source_url='/source?' + urlencode(dict(
                                   repository_root=self.root, fingerprint=self.fingerprint,
                                   relative_path=path, start_line=start, end_line=end)))
            citations.append(citation.model_copy(update=updates))
        return answer.model_copy(update={'citations': citations})
