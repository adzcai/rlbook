import{c as u}from"/build/_shared/chunk-2NH4LW52.js";var b=u((p,o)=>{function g(c){let n="\\d(_|\\d)*",s="[eE][-+]?"+n,d=n+"(\\."+n+")?("+s+")?",t="\\w+",l="\\b("+(n+"#"+t+"(\\."+t+")?#("+s+")?")+"|"+d+")",r="[A-Za-z](_?[A-Za-z0-9.])*",e=`[]\\{\\}%#'"`,a=c.COMMENT("--","$"),i={begin:"\\s+:\\s+",end:"\\s*(:=|;|\\)|=>|$)",illegal:e,contains:[{beginKeywords:"loop for declare others",endsParent:!0},{className:"keyword",beginKeywords:"not null constant access function procedure in out aliased exception"},{className:"type",begin:r,endsParent:!0,relevance:0}]};return{name:"Ada",case_insensitive:!0,keywords:{keyword:"abort else new return abs elsif not reverse abstract end accept entry select access exception of separate aliased exit or some all others subtype and for out synchronized array function overriding at tagged generic package task begin goto pragma terminate body private then if procedure type case in protected constant interface is raise use declare range delay limited record when delta loop rem while digits renames with do mod requeue xor",literal:"True False"},contains:[a,{className:"string",begin:/"/,end:/"/,contains:[{begin:/""/,relevance:0}]},{className:"string",begin:/'.'/},{className:"number",begin:l,relevance:0},{className:"symbol",begin:"'"+r},{className:"title",begin:"(\\bwith\\s+)?(\\bprivate\\s+)?\\bpackage\\s+(\\bbody\\s+)?",end:"(is|$)",keywords:"package body",excludeBegin:!0,excludeEnd:!0,illegal:e},{begin:"(\\b(with|overriding)\\s+)?\\b(function|procedure)\\s+",end:"(\\bis|\\bwith|\\brenames|\\)\\s*;)",keywords:"overriding function procedure with is renames return",returnBegin:!0,contains:[a,{className:"title",begin:"(\\bwith\\s+)?\\b(function|procedure)\\s+",end:"(\\(|\\s+|$)",excludeBegin:!0,excludeEnd:!0,illegal:e},i,{className:"type",begin:"\\breturn\\s+",end:"(\\s+|;|$)",keywords:"return",excludeBegin:!0,excludeEnd:!0,endsParent:!0,illegal:e}]},{className:"type",begin:"\\b(sub)?type\\s+",end:"\\s+",keywords:"type",excludeBegin:!0,illegal:e},i]}}o.exports=g});export default b();
